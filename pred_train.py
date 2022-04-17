import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    for i, (images, measurements) in enumerate(dataloader):
        _,_,_,_,_,lane_dist,route_angle,tl_dist,tl_state = measurements
        images = images/255
        images = images.to(device)
        lane_dist = lane_dist.to(device)
        route_angle = route_angle.to(device)
        tl_dist = tl_dist.to(device)
        tl_state = tl_state.to(device)

        outputs = model(images)
        criterion = torch.nn.L1Loss()
        criterion2 = torch.nn.BCELoss()
        tl_state = tl_state.to(torch.float32)

        loss1 = criterion(outputs[0], lane_dist)
        loss2 = criterion(outputs[1], route_angle)
        loss3 = criterion(outputs[2], tl_dist)
        loss4 = criterion2(outputs[3], tl_state)
        loss = loss1 + loss2 + loss3 + loss4
        return loss1, loss2, loss3, loss4


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    learning_rate = 0.0005
    for i, (images, measurements) in enumerate(dataloader):
        _,_,_,_,_,lane_dist,route_angle,tl_dist,tl_state = measurements
        images = images/255
        images = images.to(device)
        lane_dist = lane_dist.to(device)
        route_angle = route_angle.to(device)
        tl_dist = tl_dist.to(device)
        tl_state = tl_state.to(device)

        outputs = model(images)
        criterion = torch.nn.L1Loss()
        criterion2 = torch.nn.BCELoss()
        

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        tl_state = tl_state.to(torch.float32)
        loss1 = criterion(outputs[0], lane_dist)
        loss2 = criterion(outputs[1], route_angle)
        loss3 = criterion(outputs[2], tl_dist)
        loss4 = criterion2(outputs[3], tl_state)
        loss = loss1 + loss2 + loss3 + loss4
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss1, loss2, loss3, loss4
   


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    lane_dist_loss = []
    route_angle_loss = []
    tl_dist_loss = []
    tl_state_loss = []
    lane_dist_loss_val = []
    route_angle_loss_val = []
    tl_dist_loss_val = []
    tl_state_loss_val = []

    for i in range(10):
        lane_dist_loss.append(train_loss[i][0].item())
        route_angle_loss.append(train_loss[i][1].item())
        tl_dist_loss.append(train_loss[i][2].item())
        tl_state_loss.append(train_loss[i][3].item())
        lane_dist_loss_val.append(val_loss[i][0].item())
        route_angle_loss_val.append(val_loss[i][1].item())
        tl_dist_loss_val.append(val_loss[i][2].item())
        tl_state_loss_val.append(val_loss[i][3].item())

    y = [i+1 for i in range(10)]
    
    # plot lines
    plt.plot(y, lane_dist_loss, label = "training loss")
    plt.plot(y, lane_dist_loss_val, label = "validation loss")
    plt.legend()
    plt.title('Lane Distance')
    plt.show()

    plt.plot(y, route_angle_loss, label = "training loss")
    plt.plot(y, route_angle_loss_val, label = "validation loss")
    plt.legend()
    plt.title('Route Angle')
    plt.show()

    plt.plot(y, tl_dist_loss, label = "training loss")
    plt.plot(y, tl_dist_loss_val, label = "validation loss")
    plt.legend()
    plt.title('Trafic Light Distance')
    plt.show()

    plt.plot(y, tl_state_loss, label = "training loss")
    plt.plot(y, tl_state_loss_val, label = "validation loss")
    plt.legend()
    plt.title('Traffic Light State')
    plt.show()


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = '/home/koray/expert_data/train'
    val_root = '/home/koray/expert_data/val'
    model = AffordancePredictor()
    train_dataset = ExpertDataset(train_root, 2761)
    val_dataset = ExpertDataset(val_root, 300)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
