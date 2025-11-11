class BertClassifier(nn.Module):
	def __init__(self, freeze_bert=False):
		super(BertClassifier,self).__init__()
		D_in, H, D_out = 768, 100, 2 #H:Hidden size of our classifier
		self.bert = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-v1.1',output_hidden_states=True, output_attentions=True)
		self.num_lin_1 = nn.Linear(4,128)
		self.num_lin_2 = nn.Linear(128,768)

		self.num_lin_3 = nn.Linear(30,128)
		self.num_lin_4 = nn.Linear(128,768)

		self.classifier = nn.Sequential(
			nn.Linear(D_in,H), #replace with Attention
			nn.ReLU(),
			nn.Linear(H,D_out)
		)
		#Freeze the BERT model
		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False

	def forward(self, input_ids, attention_mask, numerical_feat_1, numerical_feat_2):
		outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		a = outputs.hidden_states
		out = a[-1]
		last_hidden_state_cls = out[:,0,:]

		num_feat_to_use = numerical_feat_1[:,:-2]
		numerical_conf_flag = numerical_feat_1[:,-2]
		numerical_sid = numerical_feat_1[:,-1]

		numerical_dx = numerical_feat_2

		num_feats = self.num_lin_1(torch.tensor(num_feat_to_use, dtype=torch.float))
		num_feats = F.relu(num_feats)
		num_feats = self.num_lin_2(num_feats)

		num_feats_dx = self.num_lin_3(numerical_dx)
		num_feats_dx = F.relu(num_feats_dx)
		num_feats_dx = self.num_lin_4(num_feats_dx)

		feed_to_clasifier = torch.add(torch.add(last_hidden_state_cls,num_feats), num_feats_dx)
		logits = self.classifier(feed_to_clasifier)

		return logits, numerical_conf_flag, numerical_sid
