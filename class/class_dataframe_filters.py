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
        - Plain numbers: "60", "60.5"
        - Negative values: "-$50.00", "-75%"
        
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
                
                # Remove currency symbols ($, €, £, ¥)
                str_value = re.sub(r'[$€£¥]', '', str_value)
                
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