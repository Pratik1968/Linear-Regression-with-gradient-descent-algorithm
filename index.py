
import torch



original_weight = 0.5
original_bias = 0.6
X = torch.arange(0,1,step=0.05)
y = original_weight*X+original_bias


def linear_function(weight,bias,x_data):
  return weight*x_data +bias

def gradient_descent_w(w,b,y_original= y,weight =True):
  y_pred = linear_function(w,b,X) 
  gradient_weight = torch.div(torch.sum(torch.mul(y_pred - y,X)),len(X))
  
  return gradient_weight

def gradient_descent_b(w,b,y_original= y,weight =True):
  y_pred = linear_function(w,b,X) 

  gradient_bias = torch.div(torch.sum(y_pred-y),len(X))
  return gradient_bias

epochs = 10000
lr =0.1
pred_w = 0.1

pred_b=0.1
for epoch in range(epochs):
  g_w = gradient_descent_w(w =pred_w, b= pred_b)
  g_b = gradient_descent_b(w=pred_w,b=pred_b)
  pred_w=pred_w - lr*g_w
  pred_b = pred_b - lr*g_b

print(f"Predicted weight :{pred_w} Predicted bias : {pred_b}")
print(f"Original weight :{original_weight} Original bias : {original_bias}")
