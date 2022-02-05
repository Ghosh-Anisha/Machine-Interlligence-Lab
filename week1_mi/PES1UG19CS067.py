#This weeks code focuses on understanding basic functions of pandas and numpy 
#This will help you complete other lab experiments


# Do not change the function definations or the parameters
import numpy as np
import pandas as pd

#input: tuple (x,y)    x,y:int 
def create_numpy_ones_array(shape):
    #return a numpy array with one at all index
    array=None
    array=np.ones(shape)
    return array

#input: tuple (x,y)    x,y:int 
def create_numpy_zeros_array(shape):
    #return a numpy array with zeros at all index
    array=None
    array=np.zeros(shape)
    return array

#input: int  
def create_identity_numpy_array(order):
	#return a identity numpy array of the defined order
    array=None
    array=np.identity(order)
    return array

#input: numpy array
def matrix_cofactor(array):
	#return cofactor matrix of the given array
	#array=None
    shape_067=array.shape
    cofactor_matrix_067=[]

    #if determinant is not zero then directly use the formula to find cofactor matrix
    """Inverse of Matrix = Adjoint of Matrix / Determinant of Matrix
        Adjoint of Matrix = Inverse of Matrix * Determinant of Matrix
        Cofactor of Matrix = transpose(Adjoint of Matrix)

        Cofactor of Matrix = transpose(Inverse of Matrix * Determinant of Matrix)
        Cofactor of Matrix = transpose(Inverse of Matrix) * Determinant of Matrix """

    if(np.linalg.det(array)!=0):
        cofactor_matrix_067=np.linalg.inv(array).T * np.linalg.det(array)

    #if it's a 2x2 matrix then directly return the cofactor
    elif shape_067[0] == 2:
        cofactor_matrix_067=np.array([[array[1][1], -1 * array[1][0]], [-1*array[0][1], array[0][0]]])

    #if it's a 1d matrix, return the original matrix itself
    elif shape_067[0] == 1:
        return array

    #if all the above cases fail then use the mathematical definition of cofactor to find the cofactor
    else:
        for row_067 in range(shape_067[0]):
            cofactor_067=[]
            for column_067 in range(shape_067[1]):
                minor_067=array[np.array(list(range(row_067))+list(range(row_067+1,shape_067[0])))[:,np.newaxis],
                        np.array(list(range(column_067))+list(range(column_067+1,shape_067[1])))]
                cofactor_value_067= ((-1)**(row_067+column_067))*np.linalg.det(minor_067)
                cofactor_067.append(cofactor_value_067)
            cofactor_matrix_067.append(cofactor_067)
    return np.array(cofactor_matrix_067)

#Input: (numpy array, int ,numpy array, int , int , int , int , tuple,tuple)
#tuple (x,y)    x,y:int 
def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):
    #note: shape is of the forst (x1,x2)
	#return W1 x (X1 ** coef1) + W2 x (X2 ** coef2) +b
	# where W1 is random matrix of shape shape1 with seed1
	# where W2 is random matrix of shape shape2 with seed2
	# where B is a random matrix of comaptible shape with seed3
	# if dimension mismatch occur return -1
    ans=None
    x1_shape_067=X1.shape
    #Raising to a power only applicable to square matrices 
    #Therefore checking if X1 and X2 are square matrices
    if(x1_shape_067[0]!=X1.shape[1]):
        return -1

    x2_shape_067=X2.shape
    if(x2_shape_067[0]!=x2_shape_067[1]):
        return -1

    #checking if multiplication is possible: r1xc1 * r2xc2 iff c1=r2
    if(shape1[1]!=x1_shape_067[0]):
        return -1

    if(shape2[1]!=x2_shape_067[0]):
        return -1

    #checking dimension of final matrices to check if addition is possible or not
    if(shape1[0]!=shape2[0] or x1_shape_067[1]!=x2_shape_067[1]):
        return -1
        
    np.random.seed(seed1)
    matrix_w1_067= np.random.rand(*shape1)
    power_1_067=np.linalg.matrix_power(X1,coef1)
    matrix_1_067=np.matmul(matrix_w1_067,power_1_067)

    np.random.seed(seed2)
    matrix_w2_067= np.random.rand(*shape2)
    power_2_067=np.linalg.matrix_power(X2,coef2)
    matrix_2_067=np.matmul(matrix_w2_067,power_2_067)

    np.random.seed(seed3)
    matrix_B_067=np.random.rand(*matrix_1_067.shape)

    ans=matrix_1_067 + matrix_2_067 + matrix_B_067
    
    return ans



def fill_with_mode(filename, column):
    """
    Fill the missing values(NaN) in a column with the mode of that column
    Args:
        filename: Name of the CSV file.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    df=None
    df=pd.read_csv(filename)
    df[column]=df[column].fillna(df[column].mode()[0])
    return df

def fill_with_group_average(df, group, column):
    """
    Fill the missing values(NaN) in column with the mean value of the 
    group the row_067 belongs to.
    The row_067s are grouped based on the values of another column

    Args:
        df: A pandas DataFrame object representing the data.
        group: The column to group the row_067s with
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    df[column]=df[column].fillna(df.groupby(group)[column].transform('mean'))

    return df

def get_rows_greater_than_avg(df, column):
    """
    Return all the rows(with all columns) where the value in a certain 'column'
    is greater than the average value of that column.

    row where row.column > mean(data.column)

    Args:
        df: A pandas DataFrame object representing the data.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        """
    df_new_067=df.loc[df[column] > df[column].mean()]
    
    return  df_new_067

