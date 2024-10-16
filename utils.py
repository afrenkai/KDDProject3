
def to_numpy(tensor):
    return tensor.cpu().detach().numpy() if tensor.requires_grad else tensor.cpu().numpy()