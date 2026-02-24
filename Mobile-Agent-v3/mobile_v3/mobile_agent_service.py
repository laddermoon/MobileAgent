"""
Mobile-Agent-v3 HTTP 服务
为 OpenClaw 提供 RESTful API 接口
"""
import os
import json
import uuid
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from utils.mobile_agent_e import InfoPool, Manager, Executor, ActionReflector, Notetaker, INPUT_KNOW
from utils.android_controller import AndroidController
from utils.harmonyos_controller import HarmonyOSController
from utils.aliyun_guiplus_wrapper import AliyunGUIPlusWrapper

load_dotenv()

app = FastAPI(title="Mobile-Agent-v3 Service", version="1.0.0")

# 全局配置
CONFIG = {
    "adb_path": os.getenv("ADB_PATH", "/opt/homebrew/bin/adb"),
    "api_key": os.getenv("APIKEY"),
    "model": "gui-plus",
    "max_steps": 25,
    "log_path": "./logs"
}

# 请求模型
class MobileTaskRequest(BaseModel):
    instruction: str
    device_type: str = "android"
    device_id: Optional[str] = None
    coor_type: str = "norm"
    max_steps: int = 20
    return_screenshots: bool = True
    timeout: int = 300

# 响应模型
class MobileTaskResponse(BaseModel):
    success: bool
    task_id: str
    instruction: str
    steps: list
    final_state: str
    execution_time: float
    error: Optional[str] = None

# 任务执行函数
def execute_mobile_task(
    instruction: str,
    device_type: str = "android",
    device_id: Optional[str] = None,
    coor_type: str = "norm",
    max_steps: int = 20,
    return_screenshots: bool = True
) -> Dict[str, Any]:
    """
    执行移动端任务
    
    Args:
        instruction: 任务指令
        device_type: 设备类型 (android/harmonyos)
        device_id: 设备 ID (可选)
        coor_type: 坐标类型 (norm/abs)
        max_steps: 最大步数
        return_screenshots: 是否返回截图
        
    Returns:
        任务执行结果
    """
    start_time = datetime.now()
    task_id = str(uuid.uuid4())
    
    try:
        # 初始化控制器
        if device_type == "android":
            controller = AndroidController(CONFIG["adb_path"])
        elif device_type == "harmonyos":
            controller = HarmonyOSController(CONFIG["adb_path"])
        else:
            raise ValueError(f"不支持的设备类型: {device_type}")
        
        # 初始化 LLM
        vllm = AliyunGUIPlusWrapper(
            api_key=CONFIG["api_key"],
            model_name=CONFIG["model"]
        )
        
        # 初始化 Agent
        info_pool = InfoPool(
            instruction=instruction,
            additional_knowledge_manager="",
            additional_knowledge_executor=INPUT_KNOW,
            err_to_manager_thresh=2
        )
        
        manager = Manager()
        executor = Executor()
        action_reflector = ActionReflector()
        notetaker = Notetaker()
        
        # 执行任务
        steps = []
        local_image_dir = None
        
        for step in range(max_steps):
            # 获取截图
            current_time = datetime.now()
            formatted_time = current_time.strftime(f'%Y-%m-%d-{current_time.hour * 3600 + current_time.minute * 60 + current_time.second}-{str(uuid.uuid4().hex[:8])}')
            
            if step == 0:
                local_image_dir = f"/tmp/screenshot_{formatted_time}.png"
            else:
                local_image_dir = local_image_dir2
            
            # 截图
            if not controller.get_screenshot(local_image_dir):
                raise Exception("获取截图失败")
            
            from PIL import Image
            width, height = Image.open(local_image_dir).size
            
            # Manager 规划
            skip_manager = False
            if len(info_pool.action_history) > 0 and info_pool.action_history[-1]['action'] == 'invalid':
                skip_manager = True
            
            if not skip_manager:
                prompt_planning = manager.get_prompt(info_pool)
                output_planning, message_manager, raw_response = vllm.predict_mm(
                    prompt_planning,
                    [local_image_dir]
                )
                
                if not raw_response:
                    raise Exception('调用 LLM 规划失败')
                
                parsed_result_planning = manager.parse_response(output_planning)
                info_pool.completed_plan = parsed_result_planning['completed_subgoal']
                info_pool.plan = parsed_result_planning['plan']
                
                # 检查是否完成
                if "Finished" in info_pool.plan.strip() and len(info_pool.plan.strip()) < 15:
                    break
            
            # Executor 执行
            prompt_action = executor.get_prompt(info_pool)
            output_action, message_operator, raw_response = vllm.predict_mm(
                prompt_action,
                [local_image_dir],
            )
            
            if not raw_response:
                raise Exception('调用 LLM 执行失败')
            
            parsed_result_action = executor.parse_response(output_action)
            action_thought = parsed_result_action['thought']
            action_object_str = parsed_result_action['action']
            action_description = parsed_result_action['description']
            
            if not action_thought or not action_object_str:
                info_pool.action_history.append({"action": "invalid"})
                info_pool.summary_history.append(action_description)
                info_pool.action_outcomes.append("C")
                info_pool.error_descriptions.append("invalid action format")
                continue
            
            action_object_str = action_object_str.replace("```", "").replace("json", "").strip()
            action_object = json.loads(action_object_str)
            
            # 检查是否回答
            if action_object['action'] == "answer":
                answer_content = action_object['text']
                break
            
            # 坐标转换
            if coor_type != "abs":
                if "coordinate" in action_object:
                    action_object['coordinate'] = [
                        int(action_object['coordinate'][0] / 1000 * width),
                        int(action_object['coordinate'][1] / 1000 * height)
                    ]
                if "coordinate2" in action_object:
                    action_object['coordinate2'] = [
                        int(action_object['coordinate2'][0] / 1000 * width),
                        int(action_object['coordinate2'][1] / 1000 * height)
                    ]
            
            # 执行物理操作
            if action_object['action'] == "click":
                controller.tap(action_object['coordinate'][0], action_object['coordinate'][1])
            elif action_object['action'] == "swipe":
                controller.slide(
                    action_object['coordinate'][0], action_object['coordinate'][1],
                    action_object['coordinate2'][0], action_object['coordinate2'][1]
                )
            elif action_object['action'] == "type":
                controller.type(action_object['text'])
            elif action_object['action'] == "back":
                controller.back()
            elif action_object['action'] == "home":
                controller.home()
            
            # 获取执行后截图
            import time
            time.sleep(1)
            local_image_dir2 = f"/tmp/screenshot_{formatted_time}_after.png"
            controller.get_screenshot(local_image_dir2)
            
            # Reflector 反思
            prompt_reflection = action_reflector.get_prompt(info_pool)
            output_reflection, message_reflector, raw_response = vllm.predict_mm(
                prompt_reflection,
                [local_image_dir, local_image_dir2],
            )
            
            parsed_result_reflection = action_reflector.parse_response(output_reflection)
            outcome = parsed_result_reflection['outcome']
            error_description = parsed_result_reflection['error_description']
            
            action_outcome = "A" if "A" in outcome else ("B" if "B" in outcome else "C")
            
            # 记录步骤
            step_info = {
                "step": step + 1,
                "action": action_object,
                "description": action_description,
                "outcome": action_outcome,
                "error": error_description if action_outcome != "A" else None
            }
            
            # 添加截图
            if return_screenshots:
                with open(local_image_dir, "rb") as f:
                    step_info["screenshot_before"] = base64.b64encode(f.read()).decode()
                with open(local_image_dir2, "rb") as f:
                    step_info["screenshot_after"] = base64.b64encode(f.read()).decode()
            
            steps.append(step_info)
            
            # 更新信息池
            info_pool.action_history.append(action_object)
            info_pool.summary_history.append(action_description)
            info_pool.action_outcomes.append(action_outcome)
            info_pool.error_descriptions.append(error_description)
        
        # 计算执行时间
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": True,
            "task_id": task_id,
            "instruction": instruction,
            "steps": steps,
            "final_state": info_pool.plan if hasattr(info_pool, 'plan') else "任务完成",
            "execution_time": execution_time,
            "error": None
        }
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        return {
            "success": False,
            "task_id": task_id,
            "instruction": instruction,
            "steps": steps if 'steps' in locals() else [],
            "final_state": "任务失败",
            "execution_time": execution_time,
            "error": str(e)
        }

@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "Mobile-Agent-v3",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "healthy"}

@app.post("/execute", response_model=MobileTaskResponse)
async def execute_task(request: MobileTaskRequest):
    """
    执行移动端任务
    
    示例:
    ```
    POST /execute
    {
        "instruction": "打开微信",
        "device_type": "android",
        "max_steps": 20
    }
    ```
    """
    try:
        result = execute_mobile_task(
            instruction=request.instruction,
            device_type=request.device_type,
            device_id=request.device_id,
            coor_type=request.coor_type,
            max_steps=request.max_steps,
            return_screenshots=request.return_screenshots
        )
        return MobileTaskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/devices")
async def list_devices():
    """列出连接的设备"""
    import subprocess
    try:
        result = subprocess.run(
            [CONFIG["adb_path"], "devices"],
            capture_output=True,
            text=True
        )
        devices = []
        for line in result.stdout.split('\n')[1:]:
            if line.strip():
                parts = line.split()
                if len(parts) >= 2:
                    devices.append({
                        "id": parts[0],
                        "status": parts[1]
                    })
        return {"devices": devices}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
