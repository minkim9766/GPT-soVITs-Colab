import os
import shutil
import subprocess
import sys

def run_command(command):
    """쉘 명령어를 실행하는 헬퍼 함수"""
    try:
        subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"오류 발생: {e}")

if sys.argv[1] == "initial":
    # 1. 필수 패키지 설치
    print("📦 필수 패키지 설치 중...")
    run_command(f"{sys.executable} -m pip install --target /content/drive/MyDrive/tts/GPT-SoVITs/package pydub faster-whisper librosa")
    run_command(f"{sys.executable} -m pip install --target /content/drive/MyDrive/tts/GPT-SoVITs/package --upgrade pip setuptools wheel gradio")
    run_command(f"{sys.executable} -m pip install --target /content/drive/MyDrive/tts/GPT-SoVITs/package opencc-python-reimplemented opencc")

    print("NOTICE:Please ReLaunch this Code!")

    import os
    os.kill(os.getpid(), 9)

# 2. Gradio frpc 파일 설정 (Linux 환경용)
elif sys.argv[1] == "install":
    import gradio
    gradio_path = os.path.dirname(gradio.__file__)
    # 시스템이 리눅스인 경우에만 frpc 다운로드 진행
    if os.name == 'posix':
        frpc_path = os.path.join(gradio_path, "frpc_linux_amd64_v0.2")
        
        if not os.path.exists(frpc_path):
            print("🌐 frpc 파일이 없어 다운로드를 시작합니다...")
            # 실제 HuggingFace 직링크를 사용해야 합니다. (아래는 예시 경로)
            run_command(f"wget -O {frpc_path} https://huggingface.co/spaces/gradio/frpc/resolve/main/frpc_linux_amd64_v0.2")
            run_command(f"chmod +x {frpc_path}")
            print("✅ frpc 설치 및 권한 설정 완료.")
        else:
            run_command(f"chmod +x {frpc_path}")
            print("✅ frpc 파일이 이미 존재합니다. 권한 재설정 완료.")

# 3. requirements.txt 설치
if os.path.exists("/content/tts/GPT-soVITs-Colab/requirements.txt"):
    print("📋 requirements.txt 설치 중...")
    run_command(f"{sys.executable} -m pip install --target /content/drive/MyDrive/tts/GPT-SoVITs/package -r requirements.txt")

# 4. 모델 웨이트(Weights) 다운로드 및 정리
target_dir = "/content/tts/GPT-soVITs-Colab/GPT_SoVITS/pretrained_models"

# 기존 폴더 삭제 (필요 시)
if os.path.exists(target_dir):
    print(f"🗑️ 기존 {target_dir} 삭제 중...")
    shutil.rmtree(target_dir)

print("📥 공식 가중치(Weights) 다운로드 중...")
run_command("git lfs install")
run_command(f"git clone https://huggingface.co/lj1995/GPT-SoVITS {target_dir}")

print("\n✨ 모든 설정이 완료되었습니다!")