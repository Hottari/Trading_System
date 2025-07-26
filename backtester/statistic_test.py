import numpy as np
import statsmodels.api as sm
from scipy import stats
from tqdm import tqdm

class StatisticTest:
    """
    A class to perform various statistical tests and analyses on financial data.
    
    Methods:
    - get_ols_results: Perform OLS regression on a DataFrame.
    - get_bootstrap_positive_return_test_ci: Calculate bootstrap confidence interval for mean returns.
    - get_one_sided_t_test_positive_return: Perform one-sample t-test for positive mean returns.
    - get_fama_macbeth_test_result: Perform Fama-MacBeth regression test.
    """
    
    def __init__(self):
        pass

    # linear ols
    def get_ols_results(self, df, dep_name, White_HC=None):
        """
        Perform OLS regression on the given DataFrame.

        Parameters:
        - df: DataFrame containing the data.
        - dep_name: Name of the dependent variable column.
        - White_HC: Optional; if provided, specifies the type of heteroskedastic, 'HC3' is White's heteroskedasticity-consistent covariance matrix estimator, better for small samples.

        Returns:
        - results: OLS regression results object.
        """
        y = df[dep_name].copy()
        X = df[[col for col in df.columns if (col!=dep_name)]].copy()
        X = sm.add_constant(X)
        model = sm.OLS(y, X)

        if White_HC is None:
            results = model.fit()
        else:
            results = model.fit(cov_type=White_HC)
        return results


    def get_bootstrap_positive_return_test_ci(self, data, n_bootstrap=100000, alpha=0.05, is_return=False):
        """
        Calculate the bootstrap confidence interval for the mean of a strategy's returns. No assumption of normality
        
        Parameters:
        - data: array-like, the returns of the strategy.
        - n_bootstrap: int, number of bootstrap samples to generate.
        - alpha: float, significance level for the confidence interval (default is 0.05).
        - is_return: bool, if True, returns a dictionary with confidence interval and significance.

        Returns:
        - results: dict, if is_return is True, contains 'ci_lower', 'ci_upper', and 'significance'.

        """
        bootstrap_means = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True).mean(axis=1)
        ci_lower = np.percentile(bootstrap_means, 100 * (alpha / 2))
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        print(f"Bootstrap {1-alpha:.0%} CI for mean strategy return: [{ci_lower:.6f}, {ci_upper:.6f}]")

        if ci_lower > 0:
            print("The strategy's mean return is significantly positive based on the CI.")
        else:
            print("The strategy's mean return is not significantly positive based on the CI.")
        
        if is_return:
            results = {
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significance': ci_lower > 0
            }
            return results


    def get_positive_return_t_test(self, data, is_return=False):
        """
        Perform a one-sample t-test to check if the mean return is significantly positive. Assumes data is normally distributed.
        
        Parameters:
        - data: pandas DataFrame or Series containing strategy returns.
        - start_out: Starting index or date for the subset of data you want to test.
        
        Returns:
        - t_stat: t-statistic for the test.
        - p_value_one_sided: One-sided p-value for testing if the mean return > 0.
        """
        
        # Perform the t-test for mean = 0
        t_stat, p_value = stats.ttest_1samp(data, 0)   # , alternative='greater')
        
        # One-sided p-value (greater than 0)
        # p_value_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        print(f"T-statistic: {t_stat:.3f}, p-value(2-sided): {p_value:.3f}")

        if is_return:
            results = {
                't_stat': t_stat,
                'p_value': p_value
            }
            return results


    def get_fama_macbeth_test_result(self, rets_data, signals_data, is_return=False):
        """
        Perform Fama-MacBeth regression to test if the selected stocks outperform others.

        Parameters:
        - rets_data: DataFrame containing returns data, indexed by time.
        - signals_data: DataFrame containing binary signals for stock selection, indexed by time.
        - is_return: bool, if True, returns a dictionary with results.

        Returns:
        - results: dict, if is_return is True, contains 'mean_return_difference', 'periods', 'p_value', and 't_statistic'.
        """
        betas = []
        for t in tqdm(signals_data.index):
            y = rets_data.loc[t]
            x = signals_data.loc[t]
            
            # Only use stocks that are not NaN in both
            mask = (np.isnan(y)==False) & (np.isnan(x)==False)
            if mask.sum() < 5:
                print(f"Skipping time {t} due to insufficient data.")
                continue 

            X = sm.add_constant(x[mask])
            y_t = y[mask]

            model = sm.OLS(y_t, X).fit()
            betas.append(model.params[1])  # coefficient on your binary signal

        # Test if mean(beta) > 0 (i.e., selected stocks outperform others)
        print('\nFama MacBeth Regression Results')
        print('----------------------------------------------------')
        print(f"Mean Return Difference (Selected - Others): {np.mean(betas):>7.5f}")
        print(f"{'Perids: '}{signals_data.shape[0]:.0f}")

        print('----------------------------------------------------')
        print('T-test for Mean beta')
        ttest_results = self.get_positive_return_t_test(data=betas, is_return=True)  # Assumes data is normally distributed

        print('----------------------------------------------------')
        print("\nBootstrap Confidence Interval for Mean beta")
        bootstrap_results = self.get_bootstrap_positive_return_test_ci(data=betas, is_return=True)

        if is_return:
            results = {
                'mean_return_difference': np.mean(betas),
                'periods': signals_data.shape[0],
                'p_value': ttest_results['p_value'],
                't_statistic': ttest_results['t_stat'],
                'ci_lower': bootstrap_results['ci_lower'],
                'ci_upper': bootstrap_results['ci_upper'],
                'significance': bootstrap_results['ci_lower'] > 0
            }
            return results
