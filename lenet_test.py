from help_lenet_test import get_model, get_features, evaluate, get_puma_model

m = get_puma_model("lenet_weights")
#m = get_model("lenet_weights")
"""
Original model:
Test loss: 0.04455459719485625
Test accuracy: 0.9871
"""
train_f , train_l, val_f, val_l, test_f, test_l = get_features()
evaluate(m, test_f, test_l)