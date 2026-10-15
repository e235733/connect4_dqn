from dqn import QNet
import torch

class TestAgent:
    def __init__(self, input_dim, output_dim):
        self.agent = QNet(input_dim, output_dim)

    def train(self, x, y):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=0.001)
        optimizer.zero_grad()
        output = self.agent.forward(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def fit(self, x, y, epochs=20):
        for epoch in range(epochs):
            loss = self.train(x, y)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        


if __name__ == "__main__":
    
    input_dim = 4
    output_dim = 2

    x = torch.randn(10, input_dim)
    y = torch.randn(10, output_dim)

    test_agent = TestAgent(input_dim, output_dim)
    test_agent.fit(x, y, epochs=20)