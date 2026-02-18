import numpy as np
import pandas as pd
from typing import Union, List, Any
import re


class DataFrameColumnFilter:
    """
    Utility class for filtering and transforming DataFrame columns.
    
    Provides methods for common data cleaning operations like type conversion,
    list length calculation, and handling null values.
    """
    
    @staticmethod
    def cast_float(column: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Convert a string numpy column to float values.
        
        Handles various formats including:
        - Currency: "$60.00", "$1,234.56"
        - Percentages: "60%", "75.5%"
        
        Args:
            column: A numpy array or pandas Series of string values
            
        Returns:
            numpy array of float values
            
        Raises:
            ValueError: If a value cannot be converted to float
            
        Examples:
            >>> import numpy as np
            >>> col = np.array(['$60.00', '$1,234.56', '75%', '45.5'])
            >>> result = DataFrameColumnFilter.cast_float(col)
            >>> print(result)
            [60.0, 1234.56, 75.0, 45.5]
        """
        # Convert to numpy array if pandas Series
        if isinstance(column, pd.Series):
            column = column.values
        
        converted = []
        
        for value in column:
            try:
                # Handle None/NaN values
                if pd.isna(value):
                    converted.append(np.nan)
                    continue
                
                # Convert to string and strip whitespace
                str_value = str(value).strip()
                
                # Remove currency symbols ($)
                str_value = re.sub(r'[$]', '', str_value)
                
                # Remove percentage sign and divide by 100 if present
                is_percentage = '%' in str_value
                str_value = str_value.replace('%', '')
                
                # Remove thousand separators (commas)
                str_value = str_value.replace(',', '')
                
                # Convert to float
                float_value = float(str_value)
                
                # Handle percentage conversion
                if is_percentage:
                    float_value = float_value / 100
                
                converted.append(float_value)
                
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Cannot convert '{value}' to float: {str(e)}")
        
        return np.array(converted, dtype=float)
    
    
    @staticmethod
    def len_list(column: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Calculate the length of each list/sequence in a column.
        
        Args:
            column: A numpy array or pandas Series containing list-like objects
            
        Returns:
            numpy array of integers representing the length of each list
            
        Raises:
            TypeError: If a value is not a list or sequence
            
        Examples:
            >>> import numpy as np
            >>> col = np.array([['a', 'b'], [1, 2, 3], ['x']], dtype=object)
            >>> result = DataFrameColumnFilter.len_list(col)
            >>> print(result)
            [2 3 1]
        """
        # Convert to numpy array if pandas Series
        if isinstance(column, pd.Series):
            column = column.values
        
        lengths = []
        
        for value in column:
            try:
                # Handle None values
                if value is None:
                    lengths.append(0)
                    continue
                
                # Handle NaN values (check after None to avoid ambiguity with lists)
                try:
                    if pd.isna(value):
                        lengths.append(0)
                        continue
                except (ValueError, TypeError):
                    # pd.isna() can fail on list-like objects, which is fine
                    pass
                
                # Calculate length
                if isinstance(value, (list, tuple, np.ndarray, str)):
                    lengths.append(len(value))
                else:
                    raise TypeError(f"Value '{value}' is not a list-like object (type: {type(value).__name__})")
                    
            except Exception as e:
                raise TypeError(f"Error processing value '{value}': {str(e)}")
        
        return np.array(lengths, dtype=int)
    
    
    @staticmethod
    def nan_to_zero(column: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Convert NaN/None values to zero in a column.
        
        Handles both numpy NaN values and Python None objects.
        Non-null values remain unchanged.
        
        Args:
            column: A numpy array or pandas Series with potential NaN/None values
            
        Returns:
            numpy array with NaN/None values replaced by 0
            
        Examples:
            >>> import numpy as np
            >>> col = np.array([1.5, np.nan, 3.0, None, 5.5], dtype=object)
            >>> result = DataFrameColumnFilter.nan_to_zero(col)
            >>> print(result)
            [1.5 0. 3. 0. 5.5]
        """
        # Convert to numpy array if pandas Series
        if isinstance(column, pd.Series):
            column = column.values.copy()
        else:
            column = column.copy()
        
        # Create mask for NaN/None values
        mask = pd.isna(column)
        
        # Replace NaN/None with 0
        column[mask] = 0
        
        return np.array(column, dtype=float)

    @staticmethod
    def cast_float(column: Union[np.ndarray, pd.Series], nan_to_zero: bool = False) -> np.ndarray:
        """
        Convert a string numpy column to float values.
        
        Handles various formats including:
        - Currency: "$60.00", "$1,234.56"
        - Percentages: "60%", "75.5%"
        - Plain numbers: "60", "60.5"
        - Negative values: "-$50.00", "-75%"
        - Nan/None values are converted to zero
        
        Args:
            column: A numpy array or pandas Series of string values
            nan_to_zero: If True, converts NaN/None values to 0. If False, leaves them as NaN.
            
        Returns:
            numpy array of float values
            
        Raises:
            ValueError: If a value cannot be converted to float
            
        Examples:
            >>> import numpy as np
            >>> col = np.array(['$60.00', '$1,234.56', '75%', '45.5'])
            >>> result = DataFrameColumnFilter.cast_float(col)
            >>> print(result)
            [60.0, 1234.56, 75.0, 45.5]

            >>> col_with_nan = np.array(['$60.00', None, '75%', np.nan])
            >>> result_nan = DataFrameColumnFilter.cast_float(col_with_nan, nan_to_zero=True)
            >>> print(result_nan)
                [60.0, 0.0, 75.0, 0.0]
        """
        # Convert to numpy array if pandas Series
        if isinstance(column, pd.Series):
            column = column.values
        
        converted = []
        
        for value in column:
            try:
                # Handle None/NaN values
                if pd.isna(value):
                    if nan_to_zero:
                        converted.append(0.0)
                    else:
                        converted.append(np.nan)
                    continue
                
                # Convert to string and strip whitespace
                str_value = str(value).strip()
                
                # Remove currency symbols ($)
                str_value = re.sub(r'[$]', '', str_value)
                
                # Remove percentage sign and divide by 100 if present
                is_percentage = '%' in str_value
                str_value = str_value.replace('%', '')
                
                # Remove thousand separators (commas)
                str_value = str_value.replace(',', '')
                
                # Convert to float
                float_value = float(str_value)
                
                # Handle percentage conversion
                if is_percentage:
                    float_value = float_value / 100
                
                converted.append(float_value)
                
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Cannot convert '{value}' to float: {str(e)}")
        
        return np.array(converted, dtype=float)
    
    
    @staticmethod
    def len_list(column: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Calculate the length of each list/sequence in a column.
        
        Args:
            column: A numpy array or pandas Series containing list-like objects
            
        Returns:
            numpy array of integers representing the length of each list
            
        Raises:
            TypeError: If a value is not a list or sequence
            
        Examples:
            >>> import numpy as np
            >>> col = np.array([['a', 'b'], [1, 2, 3], ['x']], dtype=object)
            >>> result = DataFrameColumnFilter.len_list(col)
            >>> print(result)
            [2 3 1]
        """
        # Convert to numpy array if pandas Series
        if isinstance(column, pd.Series):
            column = column.values
        
        lengths = []
        
        for value in column:
            try:
                # Handle None values
                if value is None:
                    lengths.append(0)
                    continue
                
                # Handle NaN values (check after None to avoid ambiguity with lists)
                try:
                    if pd.isna(value):
                        lengths.append(0)
                        continue
                except (ValueError, TypeError):
                    # pd.isna() can fail on list-like objects, which is fine
                    pass
                
                # Calculate length
                if isinstance(value, (list, tuple, np.ndarray, str)):
                    lengths.append(len(value))
                else:
                    raise TypeError(f"Value '{value}' is not a list-like object (type: {type(value).__name__})")
                    
            except Exception as e:
                raise TypeError(f"Error processing value '{value}': {str(e)}")
        
        return np.array(lengths, dtype=int)
    
    
    @staticmethod
    def nan_to_zero(column: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Convert NaN/None values to zero in a column.
        
        Handles both numpy NaN values and Python None objects.
        Non-null values remain unchanged.
        
        Args:
            column: A numpy array or pandas Series with potential NaN/None values
            
        Returns:
            numpy array with NaN/None values replaced by 0
            
        Examples:
            >>> import numpy as np
            >>> col = np.array([1.5, np.nan, 3.0, None, 5.5], dtype=object)
            >>> result = DataFrameColumnFilter.nan_to_zero(col)
            >>> print(result)
            [1.5 0. 3. 0. 5.5]
        """
        # Convert to numpy array if pandas Series
        if isinstance(column, pd.Series):
            column = column.values.copy()
        else:
            column = column.copy()
        
        # Create mask for NaN/None values
        mask = pd.isna(column)
        
        # Replace NaN/None with 0
        column[mask] = 0
        
        return np.array(column, dtype=float)

    @staticmethod
    def is_empty_istring(column: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Check if strings in a column are empty or contain only whitespace.
        
        Handles both numpy NaN values and Python None objects.
        Non-null values remain unchanged.
        
        Args:
            column: A numpy array or pandas Series with potential NaN/None values
            
        Returns:
            numpy array of boolean values indicating if each string is empty/whitespace
            
        Examples:
            >>> import numpy as np
            >>> col = np.array(['hello', '', '   ', 'world'])
            >>> result = DataFrameColumnFilter.is_empty_istring(col)
            >>> print(result)
            [False False  True False]
        """
        # Convert to numpy array if pandas Series
        if isinstance(column, pd.Series):
            column = column.values.copy()
        else:
            column = column.copy()
        
        # Create mask for empty/whitespace strings
        mask = []
        for value in column:
            if isinstance(value, str):
                mask.append(len(value.strip()) == 0)
            else:
                mask.append(False)  # Non-string values are not considered empty
        return np.array(mask, dtype=bool)

    @staticmethod
    def string_length(column: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Calculate the length of strings in a column.
        
        Handles both numpy NaN values and Python None objects.
        Non-string values are treated as having length 0.
        
        Args:
            column: A numpy array or pandas Series with potential NaN/None values
            
        Returns:
            numpy array of integer lengths for each string in the column
            
        Examples:
            >>> import numpy as np
            >>> col = np.array(['hello', '', '   ', 'world'])
            >>> result = DataFrameColumnFilter.string_length(col)
            >>> print(result)
            [5 0 3 5]
        """
        # Convert to numpy array if pandas Series
        if isinstance(column, pd.Series):
            column = column.values.copy()
        else:
            column = column.copy()
        
        # Create mask for string values
        lengths = []
        for value in column:
            if isinstance(value, str):
                lengths.append(len(value))
            else:
                lengths.append(0)  # Non-string values have length 0
        
        return np.array(lengths, dtype=int)

    @staticmethod
    def days_from_date(column: Union[np.ndarray, pd.Series], 
                      reference_date: Union[str, pd.Timestamp, None] = None) -> np.ndarray:
        """
        Calculate the number of days between each date in a column and a reference date.
        
        By default, calculates days from today. Can specify a custom reference date.
        Returns positive values for dates before reference (past), negative for dates after (future).
        
        Args:
            column: A numpy array or pandas Series containing date values (strings or datetime objects)
                   Supports formats like: '2014-01-22', '2014/01/22', datetime objects
            reference_date: Reference date to calculate days from (default: today's date)
                           Can be a string in format 'YYYY-MM-DD' or pd.Timestamp
            
        Returns:
            numpy array of integers representing the number of days from reference date
            
        Raises:
            ValueError: If dates cannot be parsed or invalid format
            
        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> dates = np.array(['2014-01-22', '2020-06-15', '2025-02-14'])
            >>> result = DataFrameColumnFilter.days_from_date(dates, '2025-02-14')
            >>> print(result)
            [4037  338    0]
        """
        # Convert to numpy array if pandas Series
        if isinstance(column, pd.Series):
            column = column.values
        
        # Set reference date to today if not provided
        if reference_date is None:
            reference_date = pd.Timestamp.now()
        elif isinstance(reference_date, str):
            reference_date = pd.to_datetime(reference_date)
        
        days_diff = []
        
        for value in column:
            try:
                # Handle None/NaN values
                if value is None:
                    days_diff.append(np.nan)
                    continue
                
                # Check for NaN safely
                try:
                    if pd.isna(value):
                        days_diff.append(np.nan)
                        continue
                except (ValueError, TypeError):
                    pass
                
                # Convert to datetime if string
                if isinstance(value, str):
                    date_value = pd.to_datetime(value)
                elif isinstance(value, (pd.Timestamp, np.datetime64)):
                    date_value = pd.Timestamp(value)
                else:
                    raise ValueError(f"Unsupported date type: {type(value).__name__}")
                
                # Calculate difference in days
                diff = (reference_date - date_value).days
                days_diff.append(diff)
                
            except Exception as e:
                raise ValueError(f"Cannot process date '{value}': {str(e)}")
        
        return np.array(days_diff, dtype=float)

    @staticmethod
    def cast_boolean(column: Union[np.ndarray, pd.Series], 
                    true_values: List[Any] = None,
                    false_values: List[Any] = None,
                    handle_nan: bool = False) -> np.ndarray:
        """
        Convert a column with string boolean values to actual boolean values.
        
        By default converts: 't', 'T', 'true', 'True', 'TRUE', '1', 1 → True
                            'f', 'F', 'false', 'False', 'FALSE', '0', 0 → False
        
        Args:
            column: A numpy array or pandas Series containing boolean-like values
            true_values: List of values to consider as True (default: 't', 'T', 'true', 'True', 'TRUE', '1', 1, True)
            false_values: List of values to consider as False (default: 'f', 'F', 'false', 'False', 'FALSE', '0', 0, False)
            handle_nan: If True, convert NaN/None to False; if False, keep as NaN (default: False)
            
        Returns:
            numpy array of boolean/object values (contains NaN if handle_nan=False)
            
        Raises:
            ValueError: If a value cannot be converted to boolean
            
        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> col = np.array(['t', 'f', 't', 'f', 'true', 'false'])
            >>> result = DataFrameColumnFilter.cast_boolean(col)
            >>> print(result)
            [True False True False True False]
        """
        # Convert to numpy array if pandas Series
        if isinstance(column, pd.Series):
            column = column.values
        
        # Set default true and false values
        if true_values is None:
            true_values = ['t', 'T', 'true', 'True', 'TRUE', '1', 1, True, 1.0]
        
        if false_values is None:
            false_values = ['f', 'F', 'false', 'False', 'FALSE', '0', 0, False, 0.0]
        
        booleans = []
        
        for value in column:
            try:
                # Handle None/NaN values
                if value is None:
                    if handle_nan:
                        booleans.append(False)
                    else:
                        booleans.append(np.nan)
                    continue
                
                # Check for NaN safely
                try:
                    if pd.isna(value):
                        if handle_nan:
                            booleans.append(False)
                        else:
                            booleans.append(np.nan)
                        continue
                except (ValueError, TypeError):
                    pass
                
                # Check true values
                if value in true_values:
                    booleans.append(True)
                # Check false values
                elif value in false_values:
                    booleans.append(False)
                else:
                    raise ValueError(f"Value '{value}' is not recognized as True or False")
                    
            except Exception as e:
                raise ValueError(f"Cannot convert '{value}' to boolean: {str(e)}")
        
        return np.array(booleans, dtype=object)
    
    
# # Example usage and testing
# if __name__ == "__main__":
#     # Initialize the filter class
#     df_filter = DataFrameColumnFilter()
    
#     # Example 1: Cast Float
#     print("=" * 60)
#     print("EXAMPLE 1: Cast Float")
#     print("=" * 60)
#     currency_data = np.array(['$60.00', '$1,234.56', '75%', '-$50.25', '45.5', '$0.99'])
#     print(f"Original: {currency_data}")
#     float_result = df_filter.cast_float(currency_data)
#     print(f"Converted: {float_result}")
#     print()
    
#     # Example 2: Length of Lists
#     print("=" * 60)
#     print("EXAMPLE 2: Length of Lists")
#     print("=" * 60)
#     list_data = np.array([
#         ['a', 'b', 'c'],
#         [1, 2],
#         ['x'],
#         [10, 20, 30, 40, 50]
#     ], dtype=object)
#     print(f"Original: {list_data}")
#     len_result = df_filter.len_list(list_data)
#     print(f"Lengths: {len_result}")
#     print()
    
#     # Example 3: NaN to Zero
#     print("=" * 60)
#     print("EXAMPLE 3: NaN to Zero")
#     print("=" * 60)
#     nan_data = np.array([1.5, np.nan, 3.0, None, 5.5, np.nan], dtype=object)
#     print(f"Original: {nan_data}")
#     zero_result = df_filter.nan_to_zero(nan_data)
#     print(f"With zeros: {zero_result}")
#     print()
    
#     # Example 4: Using with Pandas DataFrame
#     print("=" * 60)
#     print("EXAMPLE 4: Integration with Pandas DataFrame")
#     print("=" * 60)
#     df = pd.DataFrame({
#         'price': ['$100.00', '$2,500.50', '50%', '$75.99'],
#         'items': [['a', 'b'], [1, 2, 3], ['x'], ['p', 'q', 'r', 's']],
#         'values': [10.5, np.nan, 20.0, None]
#     })
    
#     print("Original DataFrame:")
#     print(df)
#     print()
    
#     # Apply transformations
#     df['price_float'] = df_filter.cast_float(df['price'])
#     df['item_count'] = df_filter.len_list(df['items'])
#     df['values_clean'] = df_filter.nan_to_zero(df['values'])
    
#     print("Transformed DataFrame:")
#     print(df)