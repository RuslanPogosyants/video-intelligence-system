# tests/check_gpu.py
import sys


def check_cuda():
    """Проверка доступности CUDA и GPU"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

            # Простой тест
            print("\nRunning simple GPU test...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✓ GPU computation successful")
            return True
        else:
            print("⚠ CUDA not available. Will use CPU mode.")
            return False

    except ImportError as e:
        print(f"✗ Error importing torch: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)