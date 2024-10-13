class L1_Charbonnier_entropy(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_entropy, self).__init__()
        self.lambda_l = 1
        self.lambda_h = 1
        self.eps = 1e-14

    def forward(self, X, Y):
    
        entropy_X = []
        entropy_Y = []

        cl = X[:, 0, :, :]
        cH = X[:, 1, :, :]
        cV = X[:, 2, :, :]
        cD = X[:, 3, :, :]
        coeffs = [cl, cH, cV, cD]
        energy_x = []
        for i in range(len(coeffs)):
            energy_x.append(torch.square(coeffs[i]))
        energy_total = 0
        for i in range(len(coeffs)):
            energy_total += torch.sum(energy_x[i])
        for i in range(len(coeffs)):
            energy_ratio = torch.sum(energy_x[i]) / energy_total
            entropx = -torch.sum(energy_ratio * torch.log(energy_ratio))
            entropy_X.append(entropx)

        cl = Y[:, 0, :, :]
        cH = Y[:, 1, :, :]
        cV = Y[:, 2, :, :]
        cD = Y[:, 3, :, :]
        coeffs = [cl, cH, cV, cD]
        energy = []
        for i in range(len(coeffs)):
            energy.append(torch.square(coeffs[i]))
        energy_total = 0
        for i in range(len(coeffs)):
            energy_total += torch.sum(energy[i])
        for i in range(len(coeffs)):
            energy_ratio = torch.sum(energy[i]) / energy_total
            entropy = -torch.sum(energy_ratio * torch.log(energy_ratio))
            entropy_Y.append(entropy)
        
        diff_l = torch.add(entropy_X[0], -entropy_Y[0])
        diff_h = torch.add(entropy_X[1], -entropy_Y[1])
        diff_v = torch.add(entropy_X[2], -entropy_Y[2])
        diff_d = torch.add(entropy_X[3], -entropy_Y[3])

        # print("entropy: ")
        # print(entropy_X[0], entropy_Y[0])
        # print(entropy_X[1], entropy_Y[1])
        # print(entropy_X[2], entropy_Y[2])
        # print(entropy_X[3], entropy_Y[3])

        errorl = torch.sqrt(diff_l * diff_l + self.eps) 
        errorh = torch.sqrt(diff_h * diff_h + self.eps) + torch.sqrt(diff_v * diff_v + self.eps) + torch.sqrt(diff_d * diff_d + self.eps)
        loss = self.lambda_l *errorl + self.lambda_h * errorh
        return loss
