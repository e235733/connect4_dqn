import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, input_dim, output_dim, lr=0.001, gamma=0.99):
        self.q_net = QNet(input_dim, output_dim)
        self.target_q_net = QNet(input_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.update_target_network()

    def update_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def select_action(self, state, epsilon):
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.q_net.fc3.out_features, (1,))
        else:
            with torch.no_grad():
                return torch.argmax(self.q_net(state)).unsqueeze(0)

    def train_step(self, state, action, reward, next_state, done):
        q_values = self.q_net(state)
        next_q_values = self.target_q_net(next_state)

        target_q_value = reward + (1 - done) * self.gamma * torch.max(next_q_values)

        loss = F.mse_loss(q_values.gather(1, action.unsqueeze(1)), target_q_value.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        batch = [self.buffer[idx] for idx in indices]
        state, action, reward, next_state, done = zip(*batch)
        return torch.stack(state), torch.tensor(action), torch.tensor(reward), torch.stack(next_state), torch.tensor(done)

    def __len__(self):
        return len(self.buffer)
