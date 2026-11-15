"""Face-detection guard for the October view worker (src/st9e/view_worker.py, unchanged) that stage 1 starts for the background
face box (A4 item 4(a); ACCESS_OPS2_BRIEF_02 6.70 가). Python imports this module at start-up when its folder is on
PYTHONPATH; eval5/stage1_reference.py puts it there only for that worker and sets STAGE5_FACE_GUARD_LOG. It switches OpenCL
off (cv2.ocl.setUseOpenCL(False); OPENCV_OPENCL_DEVICE=disabled is also in the environment) and sets one OpenCV thread, then
appends the worker's state (pid, cv2.ocl.useOpenCL(), cv2.getNumThreads(), OpenCV version) to that log. The GPU is hidden by
the view worker itself. Without STAGE5_FACE_GUARD_LOG it does nothing.
"""
import os

if os.environ.get('STAGE5_FACE_GUARD_LOG'):
    import cv2
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(1)
    with open(os.environ['STAGE5_FACE_GUARD_LOG'], 'a') as _f:
        _f.write(f"{os.getpid()},{int(cv2.ocl.useOpenCL())},{cv2.getNumThreads()},{cv2.__version__},"
                 f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r},{os.environ.get('OPENCV_OPENCL_DEVICE', '')}\n")
