from sklearn.metrics import accuracy_score
from SVM import CustomSVM
from Logistic_Regression_Scratch import CustomLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from DecisionForest_Scratch import SimpleDecisionForest

class ClassifierTuning:

    def decisionTreeHyperTuning(train_X, train_y, test_X, test_y):
        # Decision Forest
        num_trees = [10, 20, 30, 40]
        max_depths = [2, 3, 4, 5, 6]
        max_accuracy = 0
        optimal_tree, optimal_depth = 0,0
        for tree in num_trees:
            for depth in max_depths:
                forest = SimpleDecisionForest(num_trees=tree, max_depth=depth)
                forest.fit(train_X, train_y)
                y_pred_forest = forest.predict(test_X)
                forest_accuracy = accuracy_score(test_y, y_pred_forest)
                if forest_accuracy > max_accuracy:
                    max_accuracy = forest_accuracy
                    optimal_depth = depth
                    optimal_tree = tree

        print('Handmade Decision Tree')
        print(f'Optimal Number of Trees: {optimal_tree}')
        print(f'Optimal Tree Depth: {optimal_depth}')
        print(f'Optimal Accuracy: {round(max_accuracy * 100, 2)}')

    def handmadeLogisticalRegressionHyperTuning(train_X, train_y, test_X, test_y):
        learning_rates = [0.001, 0.01, 0.1]
        max_iterations = [50, 100, 150]
        Cs = [0, 0.1, 0.3]
        max_accuracy = 0
        optimal_rate, optimal_iterations, optimal_c = 0,0,0

        # Logistic Regression
        for c in Cs:
            for iterations in max_iterations:
                for lr in learning_rates:
                    # Create and train the model
                    logistic_regression = CustomLogisticRegression(C = c, learning_rate=lr, max_train_iterations=iterations)
                    logistic_regression.fit(train_X, train_y)

                    # Predict based on model
                    y_pred_LR = logistic_regression.predict(test_X)

                    # Find accuracy
                    lr_accuracy = accuracy_score(test_y, y_pred_LR)
                    if lr_accuracy > max_accuracy:
                        max_accuracy = lr_accuracy
                        optimal_c = c
                        optimal_rate = lr
                        optimal_iterations = iterations

        print('Handmade Logistical Regression')
        print(f'Optimal Learning Rate: {optimal_rate}')
        print(f'Optimal Number of Iterations: {optimal_iterations}')
        print(f'Optimal C value: {optimal_c}')
        print(f'Optimal Accuracy: {round(max_accuracy * 100, 2)}')


    def prebuiltLogisticalRegressionHyperTuning(train_X, train_y, test_X, test_y):
        max_iterations = [50, 100, 150]
        Cs = [0.1, 0.5, 1]
        max_accuracy = 0
        optimal_iterations, optimal_c = 0,0

        # Sci-Kit learn Logistic Regression
        for c in Cs:
            for iterations in max_iterations:
                sci_log_regression = LogisticRegression(C=c, max_iter=iterations)
                sci_log_regression.fit(train_X, train_y)

                y_pred_sci_lr = sci_log_regression.predict(test_X)

                lr_sci_accuracy = accuracy_score(test_y, y_pred_sci_lr)
                if lr_sci_accuracy > max_accuracy:
                        max_accuracy = lr_sci_accuracy
                        optimal_c = c
                        optimal_iterations = iterations


        print('Premade Logistical Regression')
        print(f'Optimal Number of Iterations: {optimal_iterations}')
        print(f'Optimal C value: {optimal_c}')
        print(f'Optimal Accuracy: {round(max_accuracy * 100, 2)}')


    def handmadeSVMHyperTuning(train_X, train_y, test_X, test_y):
        Cs = [0.001, 0.1, 1, 10, 50, 100]
        max_accuracy = 0
        optimal_c = 0

        for c in Cs:
            # Custom SVM
            svm = CustomSVM(C=c)
            svm.fit(train_X, train_y)

            # Make predictions on the test data
            y_pred_svm = svm.predict(test_X)

            # Calculate accuracy
            custom_svm_accuracy = accuracy_score(test_y, y_pred_svm)
            if custom_svm_accuracy > max_accuracy:
                        max_accuracy = custom_svm_accuracy
                        optimal_c = c

        print('Handmade SVM')
        print(f'Optimal C value: {optimal_c}')
        print(f'Optimal Accuracy: {round(max_accuracy * 100, 2)}')

    def prebuiltSVMHyperTuning(train_X, train_y, test_X, test_y):
        Cs = [0.001, 0.1, 1, 10, 50, 100]
        max_accuracy = 0
        optimal_c = 0

        for c in Cs:
            # Sci-Kit SVM
            svm = SVC(C=c)
            svm.fit(train_X, train_y)

            # Make predictions on the test data
            y_pred_svm = svm.predict(test_X)

            # Calculate accuracy
            svm_accuracy = accuracy_score(test_y, y_pred_svm)

            if svm_accuracy > max_accuracy:
                        max_accuracy = svm_accuracy
                        optimal_c = c
        print('Prebuilt SVM')
        print(f'Optimal C value: {optimal_c}')
        print(f'Optimal Accuracy: {round(max_accuracy * 100, 2)}')