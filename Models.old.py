# class MLP(nn.Module):
#     def __init__(self, number_of_features=None, learning_rate=0.001, epochs=100, save_name="", cv=5):
#         super(MLP, self).__init__()

#         self.input_dimension = number_of_features
#         self.epochs = epochs
#         self.save_path = save_name + ".pth"
#         self.learning_rate = learning_rate
#         self.cv = cv
#         self.time_taken = None

#         if number_of_features is not None:
#             self._init_model_()

#     def _init_model_(self):
#         self.model = nn.Sequential(
#             nn.Linear(self.input_dimension, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(32, 1)
#         )
#         #     nn.Linear(self.input_dimension, 128),
#         #     nn.ReLU(),
#         #     nn.Linear(128, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, 32),
#         #     nn.ReLU(),
#         #     nn.Linear(32, 1)
#         #     # No Sigmoid here because BCEWithLogitsLoss expects logits
#         # )

#         # Default loss function; might be replaced in train_epoch if class_weight is set
#         self.loss_function = nn.BCEWithLogitsLoss()
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)

#     def forward(self, x):
#         return self.model(x)

#     def train_epoch(self, X_train, y_train, class_weight=None):
#         X = torch.tensor(X_train, dtype=torch.float32).to(self.device)
#         y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)

#         # If class_weight is 'balanced', compute pos_weight and update loss function
#         if class_weight == 'balanced':
#             y_np = y_train
#             weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_np), y=y_np)
#             # pos_weight in BCEWithLogitsLoss expects the weight for the positive class
#             pos_weight = torch.tensor(weights[1] / weights[0], dtype=torch.float32).to(self.device)
#             self.loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#         else:
#             self.loss_function = nn.BCEWithLogitsLoss()

#         for epoch in range(self.epochs):
#             self.model.train()
#             self.optimizer.zero_grad()
#             out = self.forward(X)
#             loss = self.loss_function(out, y)
#             loss.backward()
#             self.optimizer.step()

#     def evaluate(self, X_val, y_val):
#         X = torch.tensor(X_val, dtype=torch.float32).to(self.device)
#         with torch.no_grad():
#             self.model.eval()
#             preds_logits = self.forward(X).cpu().numpy()
#         preds = (preds_logits > 0).astype(int)  # Threshold logits at 0 (sigmoid 0.5)
#         return f1_score(y_val, preds, average='macro')

#     def fit_save(self, X_train, y_train, param_grid=None):
#         if param_grid is None:
#             param_grid = {
#                 'learning_rate': [0.01,0.005, 0.001,0.0005],
#                 'epochs': [50, 100,150,200,250,300],
#                 'class_weight': [None, 'balanced']
#             }

#         best_score = -np.inf
#         best_params = None

#         # Grid Search (timing excluded)
#         for params in ParameterGrid(param_grid):
#             f1_scores = []

#             kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
#             for train_idx, val_idx in kf.split(X_train):
#                 X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
#                 y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

#                 self.learning_rate = params['learning_rate']
#                 self.epochs = params['epochs']
#                 class_weight = params.get('class_weight', None)
#                 self._init_model_()

#                 self.train_epoch(X_tr.to_numpy(), y_tr.to_numpy(), class_weight=class_weight)
#                 f1 = self.evaluate(X_val.to_numpy(), y_val.to_numpy())
#                 f1_scores.append(f1)

#             avg_f1 = np.mean(f1_scores)

#             if avg_f1 > best_score:
#                 best_score = avg_f1
#                 best_params = params

#         print(f"Best parameters: {best_params}")
#         print(f"Best macro F1-score (CV): {best_score:.4f}")

#         # Final training (timed)
#         self.learning_rate = best_params['learning_rate']
#         self.epochs = best_params['epochs']
#         class_weight = best_params.get('class_weight', None)
#         self._init_model_()

#         start_time = time.time()
#         self.train_epoch(X_train.to_numpy(), y_train.to_numpy(), class_weight=class_weight)
#         end_time = time.time()
#         self.time_taken = end_time - start_time

#         # Save model
#         torch.save({
#             "model": self.model.state_dict(),
#             "input_dim": self.input_dimension,
#             "time": self.time_taken
#         }, self.save_path)

#         print(f"Training time for best model: {self.time_taken:.4f} seconds")
#         print(f"Saved best model to {self.save_path}")

#     def predict_proba(self, X_test):
#         self.model.eval()
#         X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32).to(self.device)
#         with torch.no_grad():
#             logits = self.forward(X_test).cpu().numpy()
#         # probs = 1 / (1 + np.exp(-logits))  # sigmoid activation
#         probs= torch.sigmoid(torch.tensor(logits)).numpy()
#         return probs

#     def predict(self, X_test):
#         probs = self.predict_proba(X_test)
#         return (probs > 0.5).astype(int)

#     def evaluate_and_get_metrics(self, X_test, y_test):
#         y_pred = self.predict(X_test)
#         acc = accuracy_score(y_test, y_pred)

#         self.metrics = Metric(
#             accuracy=acc,
#             y_test=y_test,
#             y_pred=y_pred,
#             time_taken=self.time_taken
#         )

#         return self.metrics

#     def evaluation_mode(self, model_path):
#         model_data = torch.load(model_path)
#         self.input_dimension = model_data["input_dim"]
#         self.time_taken = model_data["time"]
#         self._init_model_()
#         self.model.load_state_dict(model_data["model"])
#         self.model.to(self.device)

# class Decision_Tree():
#     def __init__(self, random_state=42, save_name="", cv=5):
#         self.random_state = random_state
#         self.save_path = save_name + ".pkl"
#         self.cv = cv  # Number of folds for cross-validation
#         self.time_taken = None

#     def fit_save(self, X_train, y_train):
#         start_time = time.time()

#         # Define hyperparameter search space
#         param_grid = {
#             'max_depth': [3, 5, 10, 15, 20, 25, None],
#             'min_samples_split': [2, 5, 10, 20],
#             'min_samples_leaf': [1, 2, 4, 6],
#             'criterion': ['gini', 'entropy', 'log_loss'], 
#             'class_weight': [None, 'balanced'],
#             'splitter': ['best', 'random']
#         }

#         # Use F1 Macro as scoring
#         scorer = make_scorer(f1_score, average='macro')

#         # Perform Grid Search CV
#         grid_search = GridSearchCV(
#             estimator=DecisionTreeClassifier(random_state=self.random_state),
#             param_grid=param_grid,
#             scoring=scorer,
#             cv=self.cv,
#             n_jobs=-1,
#             verbose=1
#         )

#         # Fit the model
#         grid_search.fit(X_train, y_train)
#         self.model = grid_search.best_estimator_

#         end_time = time.time()
#         self.time_taken = end_time - start_time

#         print(f"Best parameters: {grid_search.best_params_}")
#         print(f"Best macro F1-score: {grid_search.best_score_}")

#         model_data = {
#             "model": self.model,
#             "time": self.time_taken
#         }

#         jlb.dump(model_data, self.save_path)
#         print(f"Saved best model to {self.save_path}")

#     def predict(self, X_test):
#         return self.model.predict(X_test)

#     def evaluate_and_get_metrics(self, X_test, y_test):
#         y_pred = self.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)

#         metrics = Metric(
#             accuracy=accuracy,
#             y_test=y_test,
#             y_pred=y_pred,
#             time_taken=self.time_taken
#         )

#         return metrics

#     def evaluation_mode(self, model_path):
#         model_data = jlb.load(model_path)
#         self.model = model_data["model"]
#         self.time_taken = model_data["time"]

# class Support_Vector_Machine():
#     def __init__(self, save_name="", cv=5):
#         self.save_path = save_name + ".pkl"
#         self.cv = cv
#         self.time_taken = None

#     def fit_save(self, X_train, y_train):
#         start_time = time.time()

#         # Define parameter grid for SVM
#         param_grid = {
#             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#             'C': [0.1, 1, 10],
#             'gamma': ['scale', 'auto']
#         }

#         scorer = make_scorer(f1_score, average='macro')

#         grid_search = GridSearchCV(
#             estimator=SVC(random_state=42),
#             param_grid=param_grid,
#             scoring=scorer,
#             cv=self.cv,
#             n_jobs=-1,
#             verbose=1
#         )

#         grid_search.fit(X_train, y_train)
#         self.model = grid_search.best_estimator_

#         end_time = time.time()
#         self.time_taken = end_time - start_time

#         print(f"Best parameters: {grid_search.best_params_}")
#         print(f"Best macro F1-score: {grid_search.best_score_:.4f}")

#         model_data = {
#             "model": self.model,
#             "time": self.time_taken
#         }

#         jlb.dump(model_data, self.save_path)
#         print(f"Saved best model to {self.save_path}")

#     def predict(self, X_test):
#         return self.model.predict(X_test)

#     def evaluate_and_get_metrics(self, X_test, y_test):
#         y_pred = self.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)

#         metrics = Metric(
#             accuracy=accuracy,
#             y_test=y_test,
#             y_pred=y_pred,
#             time_taken=self.time_taken
#         )

#         return metrics

#     def evaluation_mode(self, model_path):
#         model_data = jlb.load(model_path)
#         self.model = model_data["model"]
#         self.time_taken = model_data["time"]

# class XGBoost():
#     def __init__(self, random_state=42, save_name="", cv=5):
#         self.random_state = random_state
#         self.save_path = save_name + ".pkl"
#         self.cv = cv
#         self.time_taken = None

#     def fit_save(self, X_train, y_train):
#         start_time = time.time()

#         param_grid = {
#             'n_estimators': [50,75, 100, 150],
#             'max_depth': [3, 5, 7],
#             'learning_rate': [0.01, 0.1, 0.2],
#             'subsample': [0.7, 0.8, 1.0],
#             'colsample_bytree': [0.7, 0.8, 1.0],
#             'gamma': [0, 0.1, 0.2, 0.25, 0.3],
#             'reg_lambda': [1, 1.5, 2],
#             'reg_alpha': [0, 0.5, 1, 1.5 , 2]
#         }

#         scorer = make_scorer(f1_score, average='macro')

#         grid_search = GridSearchCV(
#             estimator=XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='mlogloss'),
#             param_grid=param_grid,
#             scoring=scorer,
#             cv=self.cv,
#             n_jobs=-1,
#             verbose=1
#         )

#         grid_search.fit(X_train, y_train)
#         self.model = grid_search.best_estimator_

#         end_time = time.time()
#         self.time_taken = end_time - start_time

#         print(f"Best parameters: {grid_search.best_params_}")
#         print(f"Best macro F1-score: {grid_search.best_score_}")

#         model_data = {
#             "model": self.model,
#             "time": self.time_taken
#         }

#         jlb.dump(model_data, self.save_path)
#         print(f"Saved best model to {self.save_path}")

#     def predict(self, X_test):
#         return self.model.predict(X_test)

#     def evaluate_and_get_metrics(self, X_test, y_test):
#         y_pred = self.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)

#         metrics = Metric(
#             accuracy=accuracy,
#             y_test=y_test,
#             y_pred=y_pred,
#             time_taken=self.time_taken
#         )

#         return metrics

#     def evaluation_mode(self, model_path):
#         model_data = jlb.load(model_path)
#         self.model = model_data["model"]
#         self.time_taken = model_data["time"]

# class K_Nearest_Neighbor():
#     def __init__(self, save_name="", cv=5):
#         self.save_path = save_name + ".pkl"
#         self.cv = cv  
#         self.time_taken = None

#     def fit_save(self, X_train, y_train):
#         start_time = time.time()

#         # Define hyperparameter grid
#         param_grid = {
#             'n_neighbors': [3, 5, 7,8, 9,10,11,12,13,14],
#             'weights': ['uniform', 'distance'],
#             'metric': ['euclidean', 'manhattan', 'minkowski']
#         }

#         # Use F1 Macro as scoring metric
#         scorer = make_scorer(f1_score, average='macro')

#         grid_search = GridSearchCV(
#             estimator=KNeighborsClassifier(),
#             param_grid=param_grid,
#             scoring=scorer,
#             cv=self.cv,
#             n_jobs=-1,
#             verbose=1
#         )

#         # Perform the grid search
#         grid_search.fit(X_train, y_train)
#         self.model = grid_search.best_estimator_

#         end_time = time.time()
#         self.time_taken = end_time - start_time

#         print(f"Best parameters: {grid_search.best_params_}")
#         print(f"Best macro F1-score: {grid_search.best_score_}")

#         model_data = {
#             "model": self.model,
#             "time": self.time_taken
#         }

#         jlb.dump(model_data, self.save_path)
#         print(f"Saved best model to {self.save_path}")

#     def predict(self, X_test):
#         return self.model.predict(X_test)

#     def evaluate_and_get_metrics(self, X_test, y_test):
#         y_pred = self.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)

#         metrics = Metric(
#             accuracy=accuracy,
#             y_test=y_test,
#             y_pred=y_pred,
#             time_taken=self.time_taken
#         )

#         return metrics

#     def evaluation_mode(self, model_path):
#         model_data = jlb.load(model_path)
#         self.model = model_data["model"]
#         self.time_taken = model_data["time"] 