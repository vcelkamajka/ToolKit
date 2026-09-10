import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pingouin import homoscedasticity
from scipy import stats
import pingouin as pg
import seaborn as sns

pd.set_option('display.max_columns', None)

# minitab but my MiniPy - use for stats

class StatisticalClass:
    def __init__(self, file_name,x_vals=None, y_vals=None, alpha = 0.05):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.file_name = file_name
        self.alpha = alpha

        self.df = pd.read_csv(self.file_name)

        pg.print_table(self.df.head(), tablefmt="grid")
        print('='*100)

        print(self.df.describe())
        print('='*100)

    def t_test_for_mean(self, group_to_test, set_val, tails = 'two-sided'):
        # checks if some mean is sig. compared to some set value

        group_vals = self.df[group_to_test]

        t_test = pg.ttest(x = group_vals, y = set_val, alternative= tails, confidence = 1 - self.alpha)

        print(f'Is there a difference between {group_to_test} mean and mean of {set_val}?:')
        pg.print_table(t_test, tablefmt="grid")
        print('='*100)


    def t_test(self, group_1, group_2, tails='two-sided'):
        # compare two groups

        group_1_vals = self.df[group_1]
        group_2_vals = self.df[group_2]

        t_test_equal_var = pg.ttest(x=group_1_vals, y=group_2_vals, alternative= tails, confidence = 1 - self.alpha)
        welch_ttest = pg.ttest(x=group_1_vals, y=group_2_vals, alternative= tails, correction=True, confidence = 1 - self.alpha)
        mwu_ttest = pg.mwu(x=group_1_vals, y=group_2_vals, alternative= tails)

        print('T-test assuming equal variance:')
        pg.print_table(t_test_equal_var, tablefmt="grid")
        print('='*100)

        print('T-test assuming unequal variance (Welch test):')
        pg.print_table(welch_ttest, tablefmt="grid")
        print('='*100)

        print('T-test assuming non-normal data (Mann-Whitney test):')
        pg.print_table(mwu_ttest, tablefmt="grid")
        print('='*100)

    def anova(self, anova_df, response, between):

        self.anova_df = pd.read_csv(anova_df)
        self.response = response
        self.between = between

        # dv: dependent variable ... response, y ...
        # between: columns to check

        for cols in self.between:
            print(f'Homoscedasticity of data for column {cols} (Levene):')
            homoscedasticity_test = pg.homoscedasticity(self.anova_df, self.response, cols, alpha = self.alpha)
            pg.print_table(homoscedasticity_test, tablefmt="grid")
        print('='*100)

        anova_test = pg.anova(self.anova_df, dv = self.response, between = self.between)

        print('ANOVA test assuming equal variance:')
        pg.print_table(anova_test, tablefmt="grid")
        print('='*100)

        if len(between) == 1:
            welch_anova = pg.welch_anova(self.anova_df, dv=self.response, between= self.between[0])
            print('ANOVA test assuming unequal variance (Welch test):')
            pg.print_table(welch_anova, tablefmt="grid")
            print('='*100)
        else:
            print('ANOVA Welch test cannot be run with more than one between group!')
            print('='*100)

        if len(between) == 1:
            tukey_anova = pg.pairwise_tukey(self.anova_df, self.response, between=self.between[0])
            print('ANOVA test to compare means of groups (Tukey HSD):')
            pg.print_table(tukey_anova, tablefmt="grid")
            print('='*100)
        else:
            print('ANOVA Tukey HSD test cannot be run with more than one between group!')
            print('='*100)

        if len(between) == 1:
            games_anova = pg.pairwise_gameshowell(self.anova_df, self.response, between=self.between[0], effsize= 'r')
            print('ANOVA test to compare means of groups (Games-Howell):')
            pg.print_table(games_anova, tablefmt="grid")
            print('='*100)
        else:
            print('ANOVA Games-Howell test cannot be run with more than one between group!')
            print('='*100)

    def normality_test(self,list_to_check):

        list_to_check = self.df[list_to_check]

        print('Normality test:')
        pg.print_table(pg.normality(list_to_check), tablefmt="grid")
        print('='*100)
        
        ax = pg.qqplot(list_to_check, dist='norm')
        plt.show()

    def bland_plot(self, group_1, group_2):

        group_1_vals = self.df[group_1]
        group_2_vals = self.df[group_2]

        ax = pg.plot_blandaltman(group_1_vals, group_2_vals, confidence = 1- self.alpha)
        plt.show()

    def distribution_plot(self, group=None, all=None):

        if all is not None:
            for n in range(len(self.df.columns)):
                plt.figure(figsize=(10, 6))
                plt.title(f'Distribution of {self.df.columns[n]}')
                sns.histplot(self.df, x=self.df[self.df.columns[n]])
                plt.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.title(f'Distribution of {group}')
            sns.histplot(self.df, x=self.df[group])
            plt.show()

    def box_plot(self, x_group, y_group):

        sns.boxplot(x = x_group, y = y_group, data = self.anova_df, color = '#569fdb', showmeans = True, meanprops={"marker": "x",
                       "markeredgecolor": "#701110",
                       "markersize": "10", 'markerfacecolor': '#701110'})

        plt.title(f'Box plot of {x_group} in regard to {y_group}')
        plt.show()



test = StatisticalClass('t_test.csv', alpha = 0.05) # - Input the file name (format: columns = different groups, rows = values for each group).
# Can also specify the significance level (alpha = 0.05 is default)

test.t_test_for_mean('Group 1', set_val = 4) # - Input column name from the filename supplied above and input a value to compare to, i.e., 4

test.t_test('Group 1', 'Group 2') # - Input two column names you wish to compare from the file supplied above

test.anova('anova_dataset.csv', response='Score',between = ['Group','Condition']) # - Input a new file to perform anova
# (format: columns = conditions and response, i.e., Group -> control, treatment, Condition -> morning, evening and response -> 54, 12 ...)
# select the response column by giving its column name and then pick column(s) such as Group and/or Condition to compare.
# NOTE: certain post-hoc tests such as Tukey HSD, Welchs and Games-Howell tests cannot be run with more than 1 column selected!

test.normality_test('Group 1') # - Runs a Shapiro-Wilk test to determine if data is normally distributed
# Input a column name from the StastisticalClass file (not the anova file)

test.bland_plot('Group 1', 'Group 3') # - Generates a Bland-Altman plot
# input two column names from the StatisticalClass file (not the anova file)

test.distribution_plot(group = 'Group 1',all=True) # - Generates a distribution plot of a given column
# input a singular column name from the StatisticalClass file - else, add 'all=True' to get the distruibution of all columns

test.box_plot('Condition', 'Score') # - Generates a boxplot for a given x and y
# input an x_group and y_group (response) from the anova file!
# NOTE: the .anova() function must be run beforehand to get the file!

