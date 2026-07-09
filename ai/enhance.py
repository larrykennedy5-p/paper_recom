import os
import json
import sys
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from queue import Queue
from threading import Lock
# INSERT_YOUR_CODE
import requests

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ai.structure import Structure
except ModuleNotFoundError:
    from structure import Structure
from recommendation import enrich_and_select

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = (Path(__file__).with_name("template.txt")).read_text(encoding="utf-8")
system = (Path(__file__).with_name("system.txt")).read_text(encoding="utf-8")


def is_sensitive(content: str) -> bool:
    """Optionally check content without making availability depend on the API.

    The upstream project called a third-party endpoint unconditionally and
    treated network errors as sensitive content. That could silently remove
    the day's only recommendation. The check is now opt-in and fail-open.
    """

    enabled = os.environ.get("ENABLE_SENSITIVE_CHECK", "false").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return False

    try:
        response = requests.post(
            "https://spam.dw-dengwei.workers.dev",
            json={"text": content},
            timeout=5,
        )
        if response.status_code == 200:
            return bool(response.json().get("sensitive", False))
        print(
            f"Sensitive check failed with status {response.status_code}; allowing content.",
            file=sys.stderr,
        )
    except Exception as error:
        print(f"Sensitive check error: {error}; allowing content.", file=sys.stderr)
    return False


def has_concrete_experiment_result(abstract: str) -> bool:
    """Conservatively detect whether an abstract states an experimental result."""

    outcome_pattern = re.compile(
        r"(results?\s+(?:show|demonstrate|indicate)|"
        r"(?:experiments?|evaluations?)\s+(?:show|demonstrate|indicate)|"
        r"outperform|surpass|"
        r"achiev(?:es|ed)\s+(?:state-of-the-art|superior|competitive|\d)|"
        r"improv(?:e|es|ed|ement).{0,30}(?:by|over|performance|accuracy|success)|"
        r"demonstrat(?:e|es|ed)\s+(?:the\s+)?(?:effectiveness|superiority)|"
        r"\b\d+(?:\.\d+)?\s*(?:%|percent|times|x)\b)",
        re.IGNORECASE,
    )
    return outcome_pattern.search(abstract or "") is not None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()

def process_single_item(chain, item: Dict, language: str) -> Dict:
    def check_github_code(content: str) -> Dict:
        """提取并验证 GitHub 链接"""
        code_info = {}

        # 1. 优先匹配 github.com/owner/repo 格式
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)
        
        if match:
            owner, repo = match.groups()
            # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
            repo = repo.rstrip(".git").rstrip(".,)")
            
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url
            
            # 尝试调用 GitHub API 获取信息
            github_token = os.environ.get("TOKEN_GITHUB")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                resp = requests.get(api_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    code_info["code_stars"] = data.get("stargazers_count", 0)
                    code_info["code_last_update"] = data.get("pushed_at", "")[:10]
            except Exception:
                # API 调用失败不影响主流程
                pass
            return code_info

        # 2. 如果没有 github.com，尝试匹配 github.io
        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
        match_io = re.search(github_io_pattern, content)
        
        if match_io:
            url = match_io.group(0)
            # 清理末尾标点
            url = url.rstrip(".,)")
            code_info["code_url"] = url
            # github.io 不进行 star 和 update 判断
                
        return code_info

    # 检查 summary 字段
    if is_sensitive(item.get("summary", "")):
        return None

    # 检测代码可用性
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    """处理单个数据项"""
    # Keep the output schema aligned with the three sections shown on the card.
    default_ai_fields = {
        "problem": "AI 总结生成失败，请根据原摘要人工核验。",
        "method": "AI 总结生成失败，请根据原摘要人工核验。",
        "experiment": "摘要中未给出充分实验细节",
    }
    
    try:
        response: Structure = chain.invoke({
            "language": "中文",
            "title": item.get("title", ""),
            "authors": "、".join(item.get("authors", [])),
            "categories": "、".join(item.get("categories", [])),
            "comment": item.get("comment") or "无",
            "content": item["summary"],
        })
        item['AI'] = response.model_dump()
    except langchain_core.exceptions.OutputParserException as e:
        # 尝试从错误信息中提取 JSON 字符串并修复
        error_msg = str(e)
        partial_data = {}
        
        if "Function Structure arguments:" in error_msg:
            try:
                # 提取 JSON 字符串
                json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
                json_str = json_str.replace('\\', '\\\\')
                # 尝试解析修复后的 JSON
                partial_data = json.loads(json_str)
            except Exception as json_e:
                print(f"Failed to parse JSON for {item.get('id', 'unknown')}: {json_e}", file=sys.stderr)
        
        # Merge partial data with defaults to ensure all fields exist
        item['AI'] = {**default_ai_fields, **partial_data}
        print(f"Using partial AI data for {item.get('id', 'unknown')}: {list(partial_data.keys())}", file=sys.stderr)
    except Exception as e:
        # Catch any other exceptions and provide default values
        print(f"Unexpected error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item['AI'] = default_ai_fields
    
    # Final validation to ensure all required fields exist
    for field in default_ai_fields.keys():
        if field not in item['AI']:
            item['AI'][field] = default_ai_fields[field]

    # A second, deterministic guard prevents an LLM from inventing results when
    # the abstract does not contain any concrete outcome signal.
    if not has_concrete_experiment_result(item.get("summary", "")):
        item["AI"]["experiment"] = "摘要中未给出充分实验细节"

    # 检查 AI 生成的所有字段
    for v in item.get("AI", {}).values():
        if is_sensitive(str(v)):
            return None
    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    llm = ChatOpenAI(
            model=model_name,
            model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}}
        ).with_structured_output(Structure, method="function_calling")

    print('Connect to:', model_name, file=sys.stderr)
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm
    
    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in enumerate(data)
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # Add default AI fields to ensure consistency
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = {
                    "problem": "AI 总结生成失败，请根据原摘要人工核验。",
                    "method": "AI 总结生成失败，请根据原摘要人工核验。",
                    "experiment": "摘要中未给出充分实验细节",
                }
    
    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    # Score every paper before any LLM call, then spend tokens only on the
    # single daily recommendation.
    data = enrich_and_select(unique_data, ROOT_DIR / "venue_whitelist.json")
    print('Open:', args.data, file=sys.stderr)
    if data:
        selected = data[0]
        print(
            "Selected:",
            selected.get("id"),
            selected.get("quality_level"),
            f"direction_score={selected.get('direction_score')}",
            file=sys.stderr,
        )
    else:
        print("No direction-relevant paper found; writing an empty recommendation file.", file=sys.stderr)

    if not data:
        Path(target_file).write_text("", encoding="utf-8")
        return
    
    # 并行处理所有数据
    processed_data = process_all_items(
        data,
        model_name,
        language,
        args.max_workers
    )
    
    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
