import torch


def alternative_decorrelation_ensemble_error(true_y, model_outputs, num_experts, num_classes, args, alternate):
    decorrelation_errors_ = []
    sse_errors = []
    # added codes just below!
    true_y = true_y.reshape(true_y.shape[0], 1)

    one_hot_target = (true_y == torch.arange(num_classes).reshape(1, num_classes).to('cuda')).float()
    for j in range(1, num_experts + 1):
        decorrelation_errors = []
        sse_error = [torch.pow((one_hot_target - model_outputs[j - 1]), 2)]

        for i in range(1, j + 1):
            if alternate == False:
                d = 1 if (i != j) else 0
            else:
                d = 1 if (i == j - 1) and i % 2 == 0 else 0

            P = (one_hot_target - model_outputs[i - 1].detach()) * (one_hot_target - model_outputs[j - 1])

            if d == 1:
                decorr_error_part = args.lambda_ * d * P
                # print('decorr_error_part:', decorr_error_part)
                decorrelation_errors.append(decorr_error_part)

        sse_errors.append(sse_error[0])
        decorrelation_errors_.extend(decorrelation_errors)

    # If we are putting all errors into one scalar and backpropagating with it!!
    decorrelation_errors_ = torch.stack(decorrelation_errors_)
    total_decorrelation_error = torch.sum(decorrelation_errors_)

    total_sse_error = torch.sum(torch.stack(sse_errors))
    total_error = total_sse_error+total_decorrelation_error
    return total_error


def wtwt_loss_update(weights, ens_out, nets_out, y, args, optimizer):
    batch_size = len(y)
    nb_digits = 4  # make sure you double check this! each dataset has different number of classes
    y_onehot = torch.FloatTensor(batch_size, nb_digits).to(args.device) # included .to(device) originally

    # # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    losses = []
    for i in range(len(nets_out)):
        net_out = nets_out[i]
        js = []
        for j in range(len(nets_out)):
            if i != j:
                netj_out = nets_out[j]
                js.append(netj_out - ens_out)
        ens_loss_i = ((net_out - ens_out) * torch.stack(js).sum(0)).mean(0)
        # weights * ((y_onehot - net_out) ** 2).mean(0) / 2
        l = (torch.tensor(weights).to(args.device) * ((y_onehot - net_out) ** 2).mean(0)/2 + ens_loss_i * 0.5).mean()
        losses.append(l)
    return torch.stack(losses).mean(0)
