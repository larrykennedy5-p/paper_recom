from pydantic import BaseModel, Field

class Structure(BaseModel):
    problem: str = Field(description="用中文具体说明论文解决的问题及其技术难点")
    method: str = Field(description="用中文具体说明论文提出的核心方法、模型或系统")
    experiment: str = Field(
        description=(
            "用中文具体说明摘要明确报告的实验设置、指标和结论；"
            "若摘要没有充分实验结论，必须原样输出“摘要中未给出充分实验细节”"
        )
    )
