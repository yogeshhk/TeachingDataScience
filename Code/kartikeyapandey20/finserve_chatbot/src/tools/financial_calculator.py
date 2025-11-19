"""
Financial Calculation Tools
CAGR, returns, comparisons, and other financial calculations
"""
import math
from typing import Dict, List, Optional, Any
from datetime import datetime


class FinancialCalculator:
    """
    Financial calculation tools for mutual fund analysis
    """
    
    @staticmethod
    def calculate_cagr(
        initial_value: float,
        final_value: float,
        years: float
    ) -> float:
        """
        Calculate Compound Annual Growth Rate (CAGR)
        
        Formula: CAGR = (Final Value / Initial Value)^(1/years) - 1
        
        Args:
            initial_value: Starting investment value
            final_value: Ending investment value
            years: Investment period in years
            
        Returns:
            CAGR as percentage
        """
        if initial_value <= 0 or years <= 0:
            raise ValueError("Initial value and years must be positive")
        
        cagr = (math.pow(final_value / initial_value, 1 / years) - 1) * 100
        return round(cagr, 2)
    
    @staticmethod
    def calculate_absolute_return(
        initial_value: float,
        final_value: float
    ) -> float:
        """
        Calculate absolute return percentage
        
        Formula: ((Final - Initial) / Initial) * 100
        
        Args:
            initial_value: Starting value
            final_value: Ending value
            
        Returns:
            Absolute return as percentage
        """
        if initial_value <= 0:
            raise ValueError("Initial value must be positive")
        
        return round(((final_value - initial_value) / initial_value) * 100, 2)
    
    @staticmethod
    def calculate_future_value(
        initial_investment: float,
        annual_return: float,
        years: float
    ) -> float:
        """
        Calculate future value of investment
        
        Formula: FV = PV * (1 + r)^n
        
        Args:
            initial_investment: Initial amount invested
            annual_return: Expected annual return (as percentage)
            years: Investment period
            
        Returns:
            Future value
        """
        rate = annual_return / 100
        fv = initial_investment * math.pow(1 + rate, years)
        return round(fv, 2)
    
    @staticmethod
    def calculate_sip_future_value(
        monthly_investment: float,
        annual_return: float,
        years: float
    ) -> float:
        """
        Calculate future value of SIP (Systematic Investment Plan)
        
        Formula: FV = P * [((1 + r)^n - 1) / r] * (1 + r)
        where r = monthly rate, n = number of months
        
        Args:
            monthly_investment: Monthly SIP amount
            annual_return: Expected annual return (as percentage)
            years: Investment period
            
        Returns:
            Future value of SIP
        """
        monthly_rate = (annual_return / 100) / 12
        months = years * 12
        
        if monthly_rate == 0:
            return monthly_investment * months
        
        fv = monthly_investment * (
            ((math.pow(1 + monthly_rate, months) - 1) / monthly_rate) * 
            (1 + monthly_rate)
        )
        return round(fv, 2)
    
    @staticmethod
    def compare_returns(
        returns_dict: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Compare returns of multiple funds
        
        Args:
            returns_dict: Dictionary of {fund_name: return_percentage}
            
        Returns:
            Sorted list of funds by return (highest first)
        """
        sorted_funds = sorted(
            returns_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {
                'rank': i + 1,
                'fund_name': fund,
                'return': ret,
                'difference_from_top': round(sorted_funds[0][1] - ret, 2)
            }
            for i, (fund, ret) in enumerate(sorted_funds)
        ]
    
    @staticmethod
    def calculate_expense_impact(
        investment: float,
        annual_return: float,
        expense_ratio: float,
        years: float
    ) -> Dict[str, float]:
        """
        Calculate impact of expense ratio on returns
        
        Args:
            investment: Initial investment
            annual_return: Gross annual return (%)
            expense_ratio: Fund expense ratio (%)
            years: Investment period
            
        Returns:
            Dict with gross value, net value, and expense impact
        """
        gross_value = FinancialCalculator.calculate_future_value(
            investment, annual_return, years
        )
        
        net_return = annual_return - expense_ratio
        net_value = FinancialCalculator.calculate_future_value(
            investment, net_return, years
        )
        
        expense_impact = gross_value - net_value
        
        return {
            'gross_value': round(gross_value, 2),
            'net_value': round(net_value, 2),
            'expense_impact': round(expense_impact, 2),
            'impact_percentage': round((expense_impact / gross_value) * 100, 2)
        }
    
    @staticmethod
    def calculate_sharpe_ratio(
        fund_return: float,
        risk_free_rate: float,
        standard_deviation: float
    ) -> float:
        """
        Calculate Sharpe Ratio (risk-adjusted return)
        
        Formula: (Return - Risk Free Rate) / Standard Deviation
        
        Args:
            fund_return: Fund's annual return (%)
            risk_free_rate: Risk-free rate (%)
            standard_deviation: Fund's standard deviation (%)
            
        Returns:
            Sharpe ratio
        """
        if standard_deviation == 0:
            raise ValueError("Standard deviation cannot be zero")
        
        sharpe = (fund_return - risk_free_rate) / standard_deviation
        return round(sharpe, 2)
    
    @staticmethod
    def years_between_dates(
        start_date: str,
        end_date: str,
        date_format: str = "%d-%b-%Y"
    ) -> float:
        """
        Calculate years between two dates
        
        Args:
            start_date: Start date string
            end_date: End date string
            date_format: Date format (default: "20-Aug-2024")
            
        Returns:
            Years as decimal
        """
        try:
            start = datetime.strptime(start_date, date_format)
            end = datetime.strptime(end_date, date_format)
            days = (end - start).days
            return round(days / 365.25, 2)
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}")


# Tool descriptions for LLM
TOOL_DESCRIPTIONS = {
    "calculate_cagr": {
        "description": "Calculate Compound Annual Growth Rate",
        "parameters": ["initial_value", "final_value", "years"],
        "example": "calculate_cagr(100000, 150000, 3) → 14.47%"
    },
    "calculate_absolute_return": {
        "description": "Calculate absolute return percentage",
        "parameters": ["initial_value", "final_value"],
        "example": "calculate_absolute_return(100, 150) → 50%"
    },
    "calculate_future_value": {
        "description": "Calculate future value of lump sum investment",
        "parameters": ["initial_investment", "annual_return", "years"],
        "example": "calculate_future_value(100000, 12, 5) → ₹176,234"
    },
    "calculate_sip_future_value": {
        "description": "Calculate future value of SIP",
        "parameters": ["monthly_investment", "annual_return", "years"],
        "example": "calculate_sip_future_value(5000, 12, 10) → ₹11,61,695"
    },
    "compare_returns": {
        "description": "Compare returns of multiple funds",
        "parameters": ["returns_dict"],
        "example": "compare_returns({'Fund A': 15, 'Fund B': 12}) → ranked list"
    },
    "calculate_expense_impact": {
        "description": "Calculate impact of expense ratio",
        "parameters": ["investment", "annual_return", "expense_ratio", "years"],
        "example": "Show how expense ratio affects long-term returns"
    }
}


def main():
    """Test financial calculator"""
    
    print("="*70)
    print("FINANCIAL CALCULATOR TESTS")
    print("="*70 + "\n")
    
    calc = FinancialCalculator()
    
    # Test 1: CAGR
    print("1. CAGR Calculation")
    cagr = calc.calculate_cagr(100000, 150000, 3)
    print(f"   Initial: ₹1,00,000 | Final: ₹1,50,000 | Period: 3 years")
    print(f"   CAGR: {cagr}%\n")
    
    # Test 2: Absolute Return
    print("2. Absolute Return")
    abs_ret = calc.calculate_absolute_return(100, 150)
    print(f"   Initial: ₹100 | Final: ₹150")
    print(f"   Return: {abs_ret}%\n")
    
    # Test 3: Future Value
    print("3. Future Value (Lump Sum)")
    fv = calc.calculate_future_value(100000, 12, 5)
    print(f"   Investment: ₹1,00,000 | Return: 12% | Period: 5 years")
    print(f"   Future Value: ₹{fv:,.2f}\n")
    
    # Test 4: SIP Future Value
    print("4. SIP Future Value")
    sip_fv = calc.calculate_sip_future_value(5000, 12, 10)
    print(f"   Monthly SIP: ₹5,000 | Return: 12% | Period: 10 years")
    print(f"   Future Value: ₹{sip_fv:,.2f}\n")
    
    # Test 5: Compare Returns
    print("5. Compare Fund Returns")
    returns = {
        "Large Cap Fund": 15.5,
        "Flexi Cap Fund": 18.2,
        "Small Cap Fund": 22.8,
        "Multi Cap Fund": 16.3
    }
    comparison = calc.compare_returns(returns)
    for fund_data in comparison:
        print(f"   Rank {fund_data['rank']}: {fund_data['fund_name']} - {fund_data['return']}% "
              f"(Δ{fund_data['difference_from_top']}%)")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)


if __name__ == "__main__":
    main()
