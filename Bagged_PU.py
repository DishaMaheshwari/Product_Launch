class PositiveUnlabeledModel(Data):

    def __init__(self, dataset, feature_columns, target_column, result_path, mask_column='mask_column', fillna_values={}, classifier=LGBMClassifier(random_state=42, n_jobs=4, importance_type='gain'), num_iterations1=50, num_iterations2=10, positive_unlabeled_ratio=0.2, data_preprocessor=None):
        self.classifier = classifier
        self.num_iterations1 = num_iterations1
        self.num_iterations2 = num_iterations2
        self.fillna_values = fillna_values
        self.feature_columns = feature_columns
        self.model_list = []
        self.positive_unlabeled_ratio = positive_unlabeled_ratio
        self.data_preprocessor = data_preprocessor
        self.result_path = result_path
        Data.__init__(self, dataset, feature_columns, target_column, mask_column)
        gc.collect()

    def save_model(self, model_object, filename):
        with open(self.result_path/filename, 'wb') as output_file:
            pickle.dump(model_object, output_file, pickle.HIGHEST_PROTOCOL)

    def remove_irrelevant_features(self, columns):
        if self.mask_column in columns:
            columns.remove(self.mask_column)
        if self.target_column in columns:
            columns.remove(self.target_column)
        if 'provider_id_upd' in columns:
            columns.remove('provider_id_upd')
        if 'predictions' in columns:
            columns.remove('predictions')
        if 'predictions2' in columns:
            columns.remove('predictions2')
        gc.collect()

        return columns

    def fill_missing_values(self, fillna_values):
        self.data.fillna(value=fillna_values, inplace=True)

    def _get_indices(self):
        positive_indices = list(self.data[self.data[self.mask_column] > 0].index)
        unlabeled_indices = list(self.data[self.data[self.mask_column] < 1].index)

        num_oob = pd.DataFrame(np.zeros(shape=self.data.shape[0]), index=self.data.index)
        sum_oob = pd.DataFrame(np.zeros(shape=self.data.shape[0]), index=self.data.index)

        gc.collect()
        return positive_indices, unlabeled_indices, num_oob, sum_oob

    def train_base_PU_model(self, train_test_split_flag=False):
        positive_indices, unlabeled_indices, num_oob, sum_oob = self._get_indices()

        self.feature_columns = self.remove_irrelevant_features(self.feature_columns)

        print("Training base model...\n")

        gc.collect()
        for iteration in range(self.num_iterations1):
            if self.data_preprocessor is not None:
                self.data = self.data_preprocessor.create_mask_var(self.data)
            
            bootstrap_sample_indices = np.random.choice(unlabeled_indices, replace=False, size=int(len(positive_indices) / self.positive_unlabeled_ratio))
            print("Positive size: ", len(positive_indices), "Unlabeled size: ", len(bootstrap_sample_indices), len(self.feature_columns))
            
            out_of_bag_indices = list(set(unlabeled_indices) - set(bootstrap_sample_indices))

            X_bootstrap = self.data[self.feature_columns][self.data[self.mask_column] > 0].append(self.data[self.feature_columns].loc[bootstrap_sample_indices])
            y_bootstrap = self.data[self.mask_column][self.data[self.mask_column] > 0].append(self.data[self.mask_column].loc[bootstrap_sample_indices])
            
            model_temp = self.classifier.fit(X_bootstrap, y_bootstrap)
            gc.collect()

            self.save_model(model_temp, "pu_model_" + str(iteration) + ".pkl")

            if iteration == 0:
                y_pred = self.classifier.predict_proba(self.data[self.feature_columns])[:, 1]
            else:
                y_pred += self.classifier.predict_proba(self.data[self.feature_columns])[:, 1]

            gc.collect()

        self.data['predictions'] = y_pred / self.num_iterations1
        gc.collect()

        return self.data
