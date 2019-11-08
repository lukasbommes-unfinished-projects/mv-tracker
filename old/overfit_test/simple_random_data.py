import torch
import torch.nn as nn


device = torch.device('cpu')

N = 64
D_in = 1000
H = 100
D_out = 10

x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)


model = nn.Sequential(
    nn.Linear(D_in, H),
    nn.ReLU(),
    nn.Linear(H, D_out)
).to(device)

criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()

for step in range(500):

    y_pred = model(x)

    loss = criterion(y_pred, y)
    print(step, loss.item())

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
