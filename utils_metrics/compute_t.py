import random, torch

def compute_t(len_t, num_timesteps, bs, device):
        all_t = torch.zeros(len_t, bs).to(device)
        idx = random.randint(0, len_t-1)
        all_t[idx] = torch.randint(0, num_timesteps, (bs,), device=device).long()

        for i in range(len_t):
            if i != idx:
                for j in range(bs):
                    all_t[i,j] = random.choice([0, num_timesteps-1, all_t[idx,j].item()])
        #print(all_t.long())
        return all_t.long()