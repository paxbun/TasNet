from model import TasNetParam

def get_param():
    return TasNetParam(N=500, L=40, H=500, K=20, C=3, g=1.5, b=0.0)

def get_directory_name(param: TasNetParam):
    return f"E:/tasnet/training_sdr_blstm_softmax_v2_6_{param.N}_{param.L}_{param.H}_{param.K}_{param.C}_{param.g}_{param.b}"