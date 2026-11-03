import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, balanced_accuracy_score,
                             f1_score, mean_absolute_error,accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier,plot_tree, export_text


class RF:
    def __init__(self, method, file, cols_to_drop = None, shap = False):
        self.method = method

        if self.method == "classifier":
            self.method = RandomForestClassifier
            type = 'classifier'
        if self.method == 'regressor':
            self.method = RandomForestRegressor
            type = 'regressor'

        self.file = file

        self.df = pd.read_csv(self.file)
        self.df.dropna(inplace=True)

        print('='*80)
        print(self.df.head())
        print('=' * 80)

        # response variable 'y' assumed to be final column - rest being x unless specified.

        if cols_to_drop is not None:
            self.df.drop(columns=cols_to_drop, inplace=True)
            print('Dropped columns:', cols_to_drop)
        all_cols = self.df.columns.tolist()

        self.x = self.df.iloc[:, 0:-1]
        self.y = self.df.iloc[:, -1]

        print(f'Response column identified to be: {self.y.name}')

        if self.method == RandomForestClassifier:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=42, stratify=self.y)
        if self.method == RandomForestRegressor:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                    random_state=42)

        self.le = LabelEncoder()
        self.hot = OneHotEncoder()

        num_cols = self.x.select_dtypes(include='number').columns.tolist()
        cat_cols = []

        for cols in all_cols:
            if self.df[cols].dtypes == 'object':
                print(f'Column: "{cols}" was found to be an object and will need to be "one hot encoded"')
                cat_cols.append(cols)

                if cols == self.y.name:
                    print(f'Column: "{cols}" was found to be an object and a response - automatic label encoding applied')

                    self.y_train = self.le.fit_transform(self.y_train)
                    self.y_test = self.le.transform(self.y_test)

                    cat_cols.remove(self.y.name)

        preprocess = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ('num', StandardScaler(), num_cols),
        ])

        pipeline = Pipeline([
            ('preprocess', preprocess),
            (type, self.method(random_state=42))
        ])

        param_grid = {f'{type}__n_estimators': [10, 30, 70, 100, 200, 300, 500],
                      f'{type}__max_depth': [None, 10, 20],
                      f'{type}__min_samples_leaf': [1, 3, 5],
                      }

        grid = GridSearchCV(pipeline, param_grid, refit=True, cv=3, n_jobs=-1)
        
        grid.fit(self.x_train, self.y_train)

        print('=' * 80)
        print(f'Best params: {grid.best_params_}')
        print(f'Best CV score: {grid.best_score_:.3f}')
        print('=' * 80)

        self.y_pred = grid.predict(self.x_test)

        model = grid.best_estimator_.named_steps[type]
        pre = grid.best_estimator_.named_steps['preprocess']
        names = [n.split('__')[-1] for n in pre.get_feature_names_out()]
        X_t = pre.transform(self.x_train)
        forest_pred = model.predict(X_t)

        Surrogate = DecisionTreeRegressor if self.method == RandomForestRegressor else DecisionTreeClassifier
        surrogate = Surrogate(max_depth=3, random_state=42).fit(X_t, forest_pred)

        fig, ax = plt.subplots(figsize=(18, 8))
        plot_tree(surrogate, feature_names=names, filled=True, rounded=True, fontsize=9, ax=ax)
        plt.show()


        if type == 'classifier':
            cm = confusion_matrix(self.y_test, self.y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')

            plt.show()

            if self.method == RandomForestRegressor:
                clf = grid.best_estimator_.named_steps['regressor']
            if self.method == RandomForestClassifier:
                clf = grid.best_estimator_.named_steps['classifier']

            best = grid.best_estimator_
            feature_names = best[:-1].get_feature_names_out()
            model = best[-1]

            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            print(feature_df)
            print('=' * 80)

            model = grid.best_estimator_.named_steps[type]
            ranked = feature_df['Feature'].str.split('__').str[-1].tolist()
            pre = grid.best_estimator_.named_steps['preprocess']

            names = [n.split('__')[-1] for n in pre.get_feature_names_out()]
            X_sample = self.x_test.sample(min(500, len(self.x_test)), random_state=42)
            X_t = pd.DataFrame(pre.transform(X_sample), columns=names, index=X_sample.index)


            if shap is True:
                explainer = shap.TreeExplainer(model)
                sv = explainer(X_t, check_additivity=False)

                sv = sv[:, :, 1]

                # global: which features matter and in which direction
                shap.plots.beeswarm(sv)

                # global: mean absolute impact (a cleaner version of feature importance)
                shap.plots.bar(sv)

                # one feature's effect, colored by its strongest interacting feature
                shap.plots.scatter(sv[:, ranked[0]], color=sv)

                # a single prediction, broken down feature by feature
                shap.plots.waterfall(sv[0])

            plt.barh(feature_df['Feature'], feature_df['Importance'])
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.xlabel('Feature Importance')
            plt.show()

            accuracy = accuracy_score(self.y_test, self.y_pred)
            #print(f'Accuracy: {accuracy:.3f}')
            print(classification_report(self.y_test, self.y_pred))

            proba = grid.predict_proba(self.x_test)

            if proba.shape[1] == 2:  # binary
                score = roc_auc_score(self.y_test, proba[:, 1])
            else:  # multiclass
                score = roc_auc_score(self.y_test, proba, multi_class="ovr")

            print(f'ROC AUC: {score:.3f}')

            print('=' * 80)


        if type == 'regressor':
            best_pipe = grid.best_estimator_
            reg = best_pipe.named_steps['regressor']
            feature_names = best_pipe.named_steps['preprocess'].get_feature_names_out()

            best = grid.best_estimator_
            feature_names = best[:-1].get_feature_names_out()
            model = best[-1]

            feature_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            print(feature_df)
            print('=' * 80)

            model = grid.best_estimator_.named_steps[type]
            ranked = feature_df['Feature'].str.split('__').str[-1].tolist()
            pre = grid.best_estimator_.named_steps['preprocess']

            # transform test data and keep readable column names
            names = [n.split('__')[-1] for n in pre.get_feature_names_out()]
            X_sample = self.x_test.sample(min(500, len(self.x_test)), random_state=42)
            X_t = pd.DataFrame(pre.transform(X_sample), columns=names, index=X_sample.index)

            if shap is True:
                explainer = shap.TreeExplainer(model)
                sv = explainer(X_t, check_additivity=False)  # returns a shap.Explanation

                # global: which features matter and in which direction
                shap.plots.beeswarm(sv)

                # global: mean absolute impact (a cleaner version of feature importance)
                shap.plots.bar(sv)

                # one feature's effect, colored by its strongest interacting feature
                shap.plots.scatter(sv[:, ranked[0]], color=sv)

                # a single prediction, broken down feature by feature
                shap.plots.waterfall(sv[0])

            top_feats = feature_df['Feature'].str.split('__').str[-1].head(4).tolist()

            plt.barh(feature_df['Feature'], feature_df['Importance'])
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.xlabel('Feature Importance')
            plt.show()

            r2 = r2_score(self.y_test, self.y_pred)
            mae = mean_absolute_error(self.y_test, self.y_pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))

            print(f'R²:   {r2:.3f}')
            print(f'MAE:  {mae:.3f}')
            print(f'RMSE: {rmse:.3f}')
            print('=' * 80)

            fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

            ax[0].scatter(self.y_test, self.y_pred, alpha=0.6)
            lims = [min(self.y_test.min(), self.y_pred.min()), max(self.y_test.max(), self.y_pred.max())]
            ax[0].plot(lims, lims, 'r--')
            ax[0].set_xlabel('Actual')
            ax[0].set_ylabel('Predicted')
            ax[0].set_title('Predicted vs Actual')

            residuals = self.y_test - self.y_pred
            ax[1].scatter(self.y_pred, residuals, alpha=0.6)
            ax[1].axhline(0, color='r', linestyle='--')
            ax[1].set_xlabel('Predicted')
            ax[1].set_ylabel('Residual (actual - predicted)')
            ax[1].set_title('Residuals')

            plt.tight_layout()
            plt.show()


#test = RF('regressor','chem_solubility_dataset.csv', ['compound_id', 'solubility_class'])
#test = RF('classifier','chem_solubility_dataset.csv', ['compound_id', 'logS'])

test = RF('regressor','chem_reaction_yield_dataset.csv', ['reaction_id', 'successful'])
test = RF('classifier','chem_reaction_yield_dataset.csv', ['reaction_id', 'yield_percent'])
