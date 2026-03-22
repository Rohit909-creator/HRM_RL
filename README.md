Psuedo Code:

out = nn.linear([state[0], state[1], action])

z_l = torch.zeros_like(out)
for i in range(h_cycle):
    z_h = GRU(out, z_l)
    z_l = z_h
    for j in range(low_cycle):
        z_l = GRU(out, z_l)

return z_h
