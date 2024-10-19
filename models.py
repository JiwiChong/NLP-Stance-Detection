import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertConfig, DistilBertModel
from torch.nn.utils.rnn import pack_padded_sequence


class MixedBertModelContextConcat(nn.Module):
    def __init__(self, ds_output, ot_output, sent_output, pre_trained='bert-base-uncased'):
        super().__init__()

        # self.bert = BertModel.from_pretrained(pre_trained)
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = self.bert.config.hidden_size
        self.LSTM = nn.LSTM(self.hidden_size, 100, bidirectional=True)
        # self.tweet = nn.Linear(self.hidden_size * 53, 100)
        self.target = nn.Linear(10, 50)
        self.ot_clf = nn.Linear(250, ot_output)

        self.distilled_bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.hidden_size_d_bert = self.distilled_bert.config.hidden_size
        self.sentiment_clf = nn.Linear(self.hidden_size_d_bert * 53, 100)
        self.sentiment = nn.Linear(100, sent_output)
        self.clf = nn.Linear(350, ds_output) # this is the layer that has concatenated target, tweet, and sentiment!

    def forward(self, i1, i2, i3, i4):
        # i4 is the context with len of 2000
        # output_bert = self.bert(input_ids=inputs[0], attention_mask=inputs[1])
             # ------------------------------- tweet part -------------------------------
        # ctx = self.ctx1(i4)
        output_bert = self.bert(input_ids=i1, attention_mask=i2)
        encoded_layers = output_bert['last_hidden_state']
        encoded_layers = encoded_layers.permute(1, 0, 2)
        enc_hiddens, (last_hidden, last_cell) = self.LSTM(pack_padded_sequence(encoded_layers, i3,
                                                                               enforce_sorted=False))
        lstm_output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        lstm_output_hidden = F.dropout(lstm_output_hidden, 0.5)
        # output_twt = self.clf(lstm_output_hidden)

        # ------------------------------- sentiment part -------------------------------
        output_distilbert = self.distilled_bert(input_ids=i1, attention_mask=i2)
        distilbert_encoded_layers = output_distilbert['last_hidden_state']
        sentiment_output = self.sentiment_clf(distilbert_encoded_layers.view(len(distilbert_encoded_layers), -1))
        sentiment_output = F.dropout(sentiment_output, 0.2)
        output_sentiment = self.sentiment(sentiment_output)


        # ------------------------------- target part -------------------------------
        target_output = self.target(i4)
        output_twt_target = torch.concat((lstm_output_hidden, target_output), axis=1)
        output_twt_target = self.ot_clf(output_twt_target)


        all_fc = torch.concat((lstm_output_hidden, sentiment_output, target_output), axis=1)
        output_stance = self.clf(all_fc)

                  # tweet to sentiment         tweet + target to opinion towards       tweet + tweet+ target to stance
        return torch.softmax(output_sentiment, 1), torch.softmax(output_twt_target, 1), torch.softmax(output_stance, 1)


class Ensemble(nn.Module):
    def __init__(self, ds_output, op_output, sent_output):
        super().__init__()
        self.nets = nn.ModuleList()
        for _ in range(3):
            self.nets.append(MixedBertModelContextConcat(ds_output, op_output, sent_output))
        # self.gate = MixedBertModel(4)

    def forward(self, i1, i2, i3, i4):
        outputs = []
        for net in self.nets:
            outputs.append(net(i1, i2, i3, i4))

        sent_outputs = [out[0] for out in outputs]
        OT_outputs = [out[1] for out in outputs]
        stance_outputs = [out[2] for out in outputs]
        # with torch.no_grad():
        ens_sent_outs = torch.stack(sent_outputs).mean(0)
        ens_OT_outs = torch.stack(OT_outputs).mean(0)
        ens_stance_outs = torch.stack(stance_outputs).mean(0)


        # ensemble output and the models output
        return ens_sent_outs, ens_OT_outs, ens_stance_outs, stance_outputs

