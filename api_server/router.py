# api_server/router.py
from fastapi import APIRouter, HTTPException
from schemas import TranslationRequest

from api_server.service import translation_service # 서비스 싱글톤 임포트

router = APIRouter()

@router.post("/translate")
def translate_sign(request: TranslationRequest):
    try:
        if not request.frames:
            raise HTTPException(status_code=400, detail="프레임 데이터가 비어 있습니다.")
            
        # 추론 서비스 호출
        translated_text = translation_service.infer(request.frames)
        
        return {
            "status": "success",
            "translated_text": translated_text  
        }
    except Exception as e:
        print(f"추론 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail="AI 추론 중 내부 서버 오류가 발생했습니다.")