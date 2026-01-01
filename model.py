import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, action_dim=7, max_joint_velocity=None):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

        self.action_dim = action_dim
        self.max_joint_velocity = max_joint_velocity

    def forward(self, x):
        output = self.layers(x)

        if self.max_joint_velocity is not None:
            #we have joint velocities w tanh and sigmoid for the binary gripper
            batch_size = output.shape[0]
            chunk_size = output.shape[1] // self.action_dim
            output_reshaped = output.view(batch_size, chunk_size, self.action_dim)

            joint_vels = output_reshaped[:, :, :6]
            gripper = output_reshaped[:, :, 6:]

            joint_vels = self.max_joint_velocity * torch.tanh(joint_vels)

            gripper = torch.sigmoid(gripper)

            output_reshaped = torch.cat([joint_vels, gripper], dim=2)
            output = output_reshaped.view(batch_size, -1)

        return output