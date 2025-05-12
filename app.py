import streamlit as st
from PIL import Image
import app3 as sudoku
import numpy as np
import cv2


def is_valid(board, row, col, num):
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    
    start_row, start_col = 3*(row // 3), 3*(col // 3)
    for i in range (3):
        for j in range(3):
            if board[start_row+i][start_col +j] == num:
                return False
            
    return True
    
def solve_sudoku(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range (1,10):
                    if is_valid(board,row,col,num):
                        board[row][col] = num
                        if solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True

def main():
    st.title("Sudoku Solver By Carlos Díez")

    with st.form("my-form"):
        sudokuImg = st.file_uploader(label="Upload your sudoku here!", accept_multiple_files=False, type=["jpg", "jpeg", "png"])
        directResoult = st.toggle(label="Resolver paso a paso", help="Por defecto se resolverá directamente a menos que se active ",disabled=True)
        submit = st.form_submit_button(label="Solve your sudoku")

        if sudokuImg and submit:
            image = Image.open(sudokuImg).convert("RGB")
            image_np = np.array(image)

            image_np_copy = image_np.copy()

            sudokuRead = sudoku.image_to_sudoku(image_np)


            col1,col2 = st.columns(2)
            with col1:
                st.image(image_np, channels="RGB", caption="Original sudoku")
            with col2:

                if solve_sudoku(sudokuRead):

                    solution = sudokuRead

                    h, w, _ = image_np_copy.shape
                    cell_w, cell_h = w // 9, h // 9
                    font = cv2.FONT_HERSHEY_SIMPLEX

                    for i in range(9):
                        for j in range(9):
                            number = str(solution[i][j])
                            x = j * cell_w + cell_w // 4
                            y = i * cell_h + 3 * cell_h // 4
                            cv2.putText(image_np_copy, number, (x, y), font, 1, (255, 0, 0), 2)

                    print(solution)
                    st.image(image_np_copy, channels="RGB", caption="Sudoku with Solution")

                    
                else:
                    st.write("Your sudoku has no solution")

if __name__ == "__main__":
    main()