import torch
import torch.nn as nn
from torch.nn.functional import max_pool2d, avg_pool2d, conv1d, max_pool1d
from pathlib import Path

torch.set_printoptions(threshold=10_000)

def write_tensor(filename, A):
    with open(filename, 'w') as file:
        for x in A.flatten():
            file.write(f"{float(x)} ")

test_dir = Path("C:/", "SGL_DATA", "05_31", "cuda_test_input")
batch = torch.tensor([[1, 2, 4, 8, 16, 12, 23, 34, 45, 56, 67], [321, 12, 34, 25, 36, 12, 23, 34, 45, 56, 67]])
wPCA = torch.tensor([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
nm = torch.tensor([42, 45])
M = 3

B = conv1d(batch.unsqueeze(1), wPCA.unsqueeze(1), padding=3//2)
Wall3 = torch.tensor([ [ [1, 2], [3, 4], [5, 6] ] , [ [7, 8], [9, 10], [11, 12] ] ])
C = torch.einsum('TkC, CkW -> TW', Wall3, B)
Cf = torch.relu(C)**2 / nm.unsqueeze(-1)
Cf[:, :M] = 0
Cf[:, -M:] = 0
Cfmax, imax = torch.max(Cf, 0)
Cmax  = max_pool1d(Cfmax.unsqueeze(0).unsqueeze(0), (2*M+1), stride = 1, padding = (M))
    
ctc_permuted = torch.tensor([ [ [ 1, 2, 3], [4, 5, 6]  ] ])
spike_templates = torch.tensor([0])
ctc_sub = ctc_permuted[spike_templates, :, :]

write_tensor(test_dir / "batch", batch)
write_tensor(test_dir / "wPCA", wPCA)
write_tensor(test_dir / "batchPCA", B)
write_tensor(test_dir / "Wall3", Wall3)
write_tensor(test_dir / "convResult", C)
write_tensor(test_dir / "nm", torch.reciprocal(nm))
write_tensor(test_dir / "Cf", Cf)
write_tensor(test_dir / "Cfmax", Cfmax)
write_tensor(test_dir / "Cmax", Cmax[0][0])
write_tensor(test_dir / "ctc_permuted", ctc_permuted)
write_tensor(test_dir / "ctc_sub", ctc_sub)
write_tensor(test_dir / "spike_templates", spike_templates)
print(f"batch.shape = {batch.shape}")
print(f"wPCA.shape = {wPCA.shape}")
print(f"batchPCA.shape = {B.shape}")
print(f"Wall3.shape = {Wall3.shape}")
print(f"convResult.shape = {C.shape}")
print(f"nm.shape = {nm.shape}")
print(f"Cf.shape = {Cf.shape}")
print(f"Cfmax.shape = {Cfmax.shape}")
print(f"Cmax.shape = {Cmax[0][0].shape}")
print(f"ctc_permuted.shape = {ctc_permuted.shape}")
print(f"spike_templates.shape = {spike_templates.shape}")
print(f"ctc_sub.shape = {ctc_sub.shape}")