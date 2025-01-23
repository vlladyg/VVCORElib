import nvidia_smi

nvidia_smi.nvmlInit()

handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
# card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

print("Total memory:", info.total/2**20)
print("Free memory:", info.free/2**20)
print("Used memory:", info.used/2**20)

nvidia_smi.nvmlShutdown()
