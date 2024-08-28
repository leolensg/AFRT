import torch

block_mask_5shot = torch.block_diag(*[torch.ones(3, 3) * -100.
                                      for _ in
                                      range(2 * 2)]).cuda()  # nk×l×l沿对角线拼接为nkl×nkl

block_mask_1shot = torch.ones(5, 5).cuda()
block_mask_1shot = (block_mask_1shot - block_mask_1shot.triu(diagonal=2) - block_mask_1shot.tril(diagonal=-2)) * -100.

print(block_mask_5shot)
print("----------------------------------------------")
print(block_mask_1shot)
