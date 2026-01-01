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

        # Apply velocity limits only if max_joint_velocity is specified
        if self.max_joint_velocity is not None:
            # Reshape to (batch, chunk_size, action_dim)
            batch_size = output.shape[0]
            chunk_size = output.shape[1] // self.action_dim
            output_reshaped = output.view(batch_size, chunk_size, self.action_dim)

            # Apply tanh to joint velocities (first 6 dimensions of each action)
            joint_vels = output_reshaped[:, :, :6]  # (batch, chunk_size, 6)
            gripper = output_reshaped[:, :, 6:]      # (batch, chunk_size, 1)

            # Scale joint velocities with tanh to bound them
            joint_vels = self.max_joint_velocity * torch.tanh(joint_vels)

            # Apply sigmoid to gripper for binary classification
            gripper = torch.sigmoid(gripper)

            # Recombine and flatten back
            output_reshaped = torch.cat([joint_vels, gripper], dim=2)
            output = output_reshaped.view(batch_size, -1)

        return output