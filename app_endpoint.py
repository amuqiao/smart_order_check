from fastapi import FastAPI
from pydantic import BaseModel
from duplication_check import WorkOrderDuplicationChecker

app = FastAPI(title="工单重复检测API")
checker = WorkOrderDuplicationChecker()

class WorkOrder(BaseModel):
    order_id: str
    content: str

class CheckRequest(BaseModel):
    content: str
    threshold: float = 0.8
    top_k: int = 5

class CheckResponse(BaseModel):
    is_duplicate: bool
    similarity: float = 0.0
    similar_order_id: str = ""
    message: str = ""

@app.post("/add_work_order", response_model=dict)
async def add_work_order(work_order: WorkOrder):
    """添加新工单到系统"""
    try:
        checker.add_work_order(work_order.order_id, work_order.content)
        return {"status": "success", "message": f"工单 {work_order.order_id} 已添加"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/check_duplication", response_model=CheckResponse)
async def check_duplication(request: CheckRequest):
    """检查工单是否重复"""
    duplicates = checker.check_duplication(
        request.content, 
        top_k=request.top_k, 
        threshold=request.threshold
    )
    
    if duplicates:
        return CheckResponse(
            is_duplicate=True,
            similarity=duplicates[0][1],
            similar_order_id=duplicates[0][0],
            message=f"发现相似工单，相似度: {duplicates[0][1]:.2f}"
        )
    else:
        return CheckResponse(
            is_duplicate=False,
            message="未发现重复工单"
        )

# 启动命令: uvicorn api_endpoint:app --reload    