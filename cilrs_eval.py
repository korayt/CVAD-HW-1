import os
import torch
import yaml
from torchvision import transforms
from carla_env.env import Env
import numpy as np
import PIL
def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = tensor.permute(1,2,0)
    tensor = np.array(tensor, dtype=np.uint8)
    return PIL.Image.fromarray(tensor)

class Evaluator():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = self.load_agent()
        self.transformer = transforms.Compose([transforms.Resize((224, 224))])

    def load_agent(self):
        return torch.load("cilrs_model.ckpt")


    def generate_action(self, rgb, command, speed):
        rgb = rgb/255
        rgb = torch.from_numpy(rgb)
        rgb = rgb.float()
        rgb = rgb.permute(2, 0, 1)
        rgb = self.transformer(rgb)
        #tensor_to_image(rgb).show()
        rgb = rgb.unsqueeze(0)
        return self.agent(rgb, command, speed)

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, brake, steer = self.generate_action(rgb, command, speed)
        throttle = float(throttle)
        brake = float(brake)
        steer = float(steer)
        print('throttle: ' + str(throttle))
        print('brake: ' + str(brake))
        print('steer: ' + str(steer))
        #throttle += 1
        #brake = 0
        
        action = {
            "throttle": throttle,
            "brake": brake,
            "steer": steer
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0)+1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
