import os
import random
import numpy as np

def innerProduct(a, b):
    sum=0
    for i in range(0, 5):
        for j in range(0, 5):
            sum += a[i][j]*b[i][j]
    return sum

class Block:
    def __init__(self):
        self.blockBoard = np.zeros((5, 5), dtype=np.int)
        self.blockList = [[(2, 2)],
                [(2, 2), (2, 3)], [(2, 2), (1, 2)], [(1, 1), (2, 2)], [(3, 1), (2, 2)],
                [(2, 1), (2, 2), (2, 3)], [(1, 2), (2, 2), (3, 2)], [(1, 1), (2, 2), (3, 3)], [(2, 2), (2, 3), (1, 3)],
                [(1, 2), (2, 2), (1, 3)], [(3, 1), (2, 2), (1, 3)], [(2, 2), (3, 2), (2, 3)], [(1, 2), (2, 2), (2, 3)],
                [(1, 1), (2, 2), (3, 3), (4, 4)], [(4, 0), (3, 1), (2, 2), (1, 3)], [(2, 1), (2, 2), (2, 3), (2, 4)], [(1, 2), (2, 2), (3, 2), (4, 2)], 
                [(2, 2), (1, 3), (2, 3), (3, 3)], [(2, 2), (3, 1), (3, 2), (3, 3)], [(1, 1), (1, 2), (2, 2), (1, 3)], [(1, 1), (2, 1), (2, 2), (3, 1)], 
                [(2, 2), (3, 2), (1, 3), (2, 3)], [(2, 1), (2, 2), (3, 2), (3, 3)], [(3, 1), (3, 2), (2, 2), (2, 3)], [(1, 1), (2, 1), (2, 2), (3, 2)], 
                [(2, 1), (2, 2), (2, 3), (1, 3)], [(1, 1), (2, 1), (2, 2), (2, 3)], [(2, 1), (2, 2), (2, 3), (3, 3)], [(2, 1), (2, 2), (2, 3), (3, 1)],
                [(3, 1), (3, 2), (2, 2), (1, 2)], [(1, 2), (2, 2), (3, 2), (3, 3)], [(1, 1), (1, 2), (2, 2), (3, 2)], [(1, 2), (1, 3), (2, 2), (3, 2)], 
                [(1, 1), (1, 2), (2, 1), (2, 2)], 
                [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)], [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
                [(1, 1), (2, 1), (2, 2), (1, 3), (2, 3)], [(2, 1), (2, 2), (2, 3), (3, 1), (3, 3)], [(1, 1), (1, 2), (2, 2), (3, 1), (3, 2)], [(1, 2), (1, 3), (2, 2), (3, 2), (3, 3)],
                [(1, 1), (2, 1), (3, 1), (3, 2), (3, 3)], [(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)], [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)], [(3, 1), (3, 2), (3, 3), (2, 3), (1, 3)], 
                [(1, 2), (2, 2), (3, 1), (3, 2), (3, 3)], [(1, 1), (2, 1), (2, 2), (2, 3), (3, 1)], [(2, 1), (2, 2), (2, 3), (1, 3), (3, 3)], [(1, 1), (1, 2), (1, 3), (2, 2), (3, 2)],
                [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2)], []]
        
    def sample(self, index):
        for i in self.blockList[index]:
            self.blockBoard[i] += 1
        
class Board:
    def __init__(self):
        self.reset()
        self.action_space = list(range(243))
        self.observation = np.zeros(84)
        
    def reset(self):
        self.indexes = [49, 49, 49]
        
        self.center = np.zeros((9, 9), dtype=np.int)
        self.vertical = np.ones((9, 2), dtype=np.int)
        self.horizontal = np.ones((2, 13), dtype=np.int)

        self.board = np.concatenate((self.horizontal, np.concatenate((self.vertical, self.center, self.vertical), axis=1), self.horizontal), axis=0)
        self.blocks = self.setBlocks()
        self.done = False
        self.reward = 0  
        
        return np.zeros(84)

    def observationTranslate(self):
        for i in range(9):
            self.observation[i] = self.board[2][i+2]
            self.observation[9+i] = self.board[3][i+2]
            self.observation[18+i] = self.board[4][+2]
            self.observation[27+i] = self.board[5][i+2]
            self.observation[36+i] = self.board[6][i+2]
            self.observation[45+i] = self.board[7][i+2]
            self.observation[54+i] = self.board[8][i+2]
            self.observation[63+i] = self.board[9][i+2]
            self.observation[72+i] = self.board[10][i+2]            
        self.observation[81] = self.indexes[0]
        self.observation[82] = self.indexes[1]
        self.observation[83] = self.indexes[2]
        
        return self.observation
    
    def actionTranslate(self, a):
        if a // 81 == 0:
            k = 0
            if a // 9 == 0:
                i = 0
                j = a
            if a // 9 == 1:
                i = 1
                j = a-9
            if a // 9 == 2:
                i = 2
                j = a-18
            if a // 9 == 3:
                i = 3
                j = a-27
            if a // 9 == 4:
                i = 4
                j = a-36
            if a // 9 == 5:
                i = 5
                j = a-45
            if a // 9 == 6:
                i = 6
                j = a-54
            if a // 9 == 7:
                i = 7
                j = a-63
            if a // 9 == 8:
                i = 8
                j = a-72

        if a // 81 == 1:
            k = 1
            if (a-81) // 9 == 0:
                i = 0
                j = a-81
            if (a-81) // 9 == 1:
                i = 1
                j = a-9-81
            if (a-81) // 9 == 2:
                i = 2
                j = a-18-81
            if (a-81) // 9 == 3:
                i = 3
                j = a-27-81
            if (a-81) // 9 == 4:
                i = 4
                j = a-36-81
            if (a-81) // 9 == 5:
                i = 5
                j = a-45-81
            if (a-81) // 9 == 6:
                i = 6
                j = a-54-81
            if (a-81) // 9 == 7:
                i = 7
                j = a-63-81
            if (a-81) // 9 == 8:
                i = 8
                j = a-72-81    

        if a // 81 == 2:
            k = 2
            if (a-162) // 9 == 0:
                i = 0
                j = a-162
            if (a-162) // 9 == 1:
                i = 1
                j = a-9-162
            if (a-162) // 9 == 2:
                i = 2
                j = a-18-162
            if (a-162) // 9 == 3:
                i = 3
                j = a-27-162
            if (a-162) // 9 == 4:
                i = 4
                j = a-36-162
            if (a-162) // 9 == 5:
                i = 5
                j = a-45-162
            if (a-162) // 9 == 6:
                i = 6
                j = a-54-162
            if (a-162) // 9 == 7:
                i = 7
                j = a-63-162
            if (a-162) // 9 == 8:
                i = 8
                j = a-72-162

        action = k, i, j
        return action
        
    def setBlocks(self):
        self.block_remain = 3
        
        self.indexes = random.sample(range(0, 49), 3)
        self.indexes.sort()
        
        givenBlocks=[]

        block0=Block()
        block0.sample(self.indexes[0])
        givenBlocks.append(block0.blockBoard)

        block1=Block()
        block1.sample(self.indexes[1])
        givenBlocks.append(block1.blockBoard)

        block2=Block()
        block2.sample(self.indexes[2])
        givenBlocks.append(block2.blockBoard)
        
        return givenBlocks
    
    def render(self):
        print('board\n', self.board, '\n')
        for i in range(0, len(self.blocks)):
            print('block', i, '\n', self.blocks[i], '\n')

    def step(self, action):
        k, i, j = self.actionTranslate(action)
                
        if k not in range(3):
            self.reward = -27
            self.observation = self.observationTranslate()
            self.done = True
            return self.observation, self.reward, self.done
        
        if i not in range(9):
            self.reward = -27
            self.observation = self.observationTranslate()
            self.done = True
            return self.observation, self.reward, self.done
            
        if j not in range(9):
            self.reward = -27
            self.observation = self.observationTranslate()
            self.done = True
            return self.observation, self.reward, self.done

        if self.indexes[k] == 49:
            self.reward = -9
            self.observation = self.observationTranslate()
            return self.observation, self.reward, self.done
        
        if self.noActionPossible(self.blocks) != False:
            self.done = True
            
        if self.feasibleTest(action) == False:
            self.observation = self.observationTranslate()
            return self.observation, self.reward, self.done
        
        else:
            self.board[i:i+5, j:j+5] += self.blocks[k]
            self.blocks[k] = np.zeros((5, 5), dtype=np.int)
            self.indexes[k] = 49
            self.block_remain -= 1
            self.reward = self.completed()
    
        if self.block_remain == 0:
            self.blocks = self.setBlocks()

        self.observation = self.observationTranslate()
        return self.observation, self.reward, self.done
    
    def feasibleTest(self, action):
        k, i, j = self.actionTranslate(action)
        boardPiece = np.zeros((5, 5), dtype=np.int)

        for m in range(0, 5):
            for n in range(0, 5):
                boardPiece[n][m]=self.board[i+n][j+m]
        
        if innerProduct(boardPiece, self.blocks[k])==0:
            return True
        else:
            return False
    
    def noActionPossible(self, blocks):
        for k in range(0, len(blocks)):
            for i in range(0, 9):
                for j in range(0, 9):
                    action = k*81 + i*9 + j
                    if self.feasibleTest(action) == True:
                        return False
                        break
        
    def completed(self):
        reward = 3

        row = np.zeros(9, dtype=np.int)
        col = np.zeros(9, dtype=np.int)
        box = np.zeros(9, dtype=np.int)

        rowFull=[]
        colFull=[]
        boxFull=[]
        
        boxVertex=[(2, 2), (2, 5), (2, 8), (5, 2), (5, 5), (5, 8), (8, 2), (8, 5), (8, 8)]

        for i in range(0, 9):
            row[i]=self.board[i+2][2]+self.board[i+2][3]+self.board[i+2][4]+self.board[i+2][5]+self.board[i+2][6]+self.board[i+2][7]+self.board[i+2][8]+self.board[i+2][9]+self.board[i+2][10]
            col[i]=self.board[2][i+2]+self.board[3][i+2]+self.board[4][i+2]+self.board[5][i+2]+self.board[6][i+2]+self.board[7][i+2]+self.board[8][i+2]+self.board[9][i+2]+self.board[10][i+2]
            box[i]=self.board[boxVertex[i][0]][boxVertex[i][1]]+self.board[boxVertex[i][0]+1][boxVertex[i][1]]+self.board[boxVertex[i][0]][boxVertex[i][1]+1]+self.board[boxVertex[i][0]+1][boxVertex[i][1]+1]+self.board[boxVertex[i][0]+2][boxVertex[i][1]+1]+self.board[boxVertex[i][0]+1][boxVertex[i][1]+2]+self.board[boxVertex[i][0]+2][boxVertex[i][1]]+self.board[boxVertex[i][0]][boxVertex[i][1]+2]+self.board[boxVertex[i][0]+2][boxVertex[i][1]+2]
        
        for i in range(0, 9):
            if row[i]==9:
                reward+=18
                rowFull.append(i)
            if col[i]==9:
                reward+=18
                colFull.append(i)
            if box[i]==9:
                reward+=18
                boxFull.append(i)
        
        for i in rowFull:
            for j in range(2, 11):
                self.board[i+2][j]=0

        for i in colFull:
            for j in range(2, 11):
                self.board[j][i+2]=0

        for i in boxFull:
            self.board[boxVertex[i][0]][boxVertex[i][1]]=0
            self.board[boxVertex[i][0]+1][boxVertex[i][1]]=0
            self.board[boxVertex[i][0]][boxVertex[i][1]+1]=0
            self.board[boxVertex[i][0]+1][boxVertex[i][1]+1]=0
            self.board[boxVertex[i][0]+2][boxVertex[i][1]+1]=0
            self.board[boxVertex[i][0]+1][boxVertex[i][1]+2]=0
            self.board[boxVertex[i][0]+2][boxVertex[i][1]]=0
            self.board[boxVertex[i][0]][boxVertex[i][1]+2]=0
            self.board[boxVertex[i][0]+2][boxVertex[i][1]+2]=0
        
        return reward