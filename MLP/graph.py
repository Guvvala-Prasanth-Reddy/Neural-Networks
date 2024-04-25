from torchview import draw_graph
from torchviz import make_dot


def draw_graph(model , batch_size , input_size):
# device='meta' -> no memory is consumed for visualization
    model_graph = draw_graph(model, input_size=(batch_size, input_size), device='meta')
    model_graph.visual_graph

def gradient_propogation(model , test_loader):
    X,Y = next(iter(test_loader))
    device = next(model.parameters()).device
    X = X.to(device)
    yhat = model(X)

    make_dot(yhat.mean(), params=dict(model.named_parameters()))