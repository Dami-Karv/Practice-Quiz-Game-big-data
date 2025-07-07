# hy485 Quiz app

A comprehensive quiz application for machine learning topics including NumPy, pandas, neural networks, CNNs, RNNs, transformers, and more.

## Features

### Quiz Types
- **Quiz #1**: NumPy, pandas, matplotlib, statistics (41 questions)
- **Quiz #2**: PCA, neural networks, ML algorithms (41 questions)  
- **Quiz #3**: CNNs, RNNs, transformers, vision (20 questions)
- **Quiz #4**: NLP, video analysis, reinforcement learning (22 questions)
- **Everything Test**: All questions from all quizzes combined
- **Random Quiz**: Custom number of randomly selected questions

### Question Types
- **Multiple Choice**: A, B, C, D options with randomized order
- **Input/Calculation**: Text input for numerical or short answers
- **True/False**: Binary choice questions

### Question Randomization
- **Option Shuffling**: A, B, C, D choices are randomized for each multiple choice question
- **Question Order**: Optional randomization of question sequence
- **Everything Test**: Always randomized automatically
- **Answer Tracking**: Correct answers tracked regardless of shuffled positions

### Quiz Configuration
- **Question Type Filtering**: Select/deselect Multiple Choice, Calculations, True/False
- **Randomization Control**: Toggle question order randomization for individual quizzes
- **Flexible Setup**: Take full quizzes or filtered subsets

### Navigation & Controls
- **Forward-Only Progression**: No back button, questions advance sequentially
- **Keyboard Shortcuts**: Space bar to submit/continue answers
- **Menu Access**: Return to main menu from any question
- **Restart Functionality**: Restart current quiz with same configuration

### Progress Tracking
- **Circular Progress Indicator**: Visual progress with correct/incorrect breakdown
- **Question Counter**: Current question number and total count
- **Real-time Updates**: Progress updates after each answer

### Timer Features
- **Everything Test Timer**: Automatic timing for Everything Test
- **Live Display**: Real-time timer display during quiz
- **Final Time**: Time shown in results for Everything Test

### Answer Feedback
- **Immediate Feedback**: Correct/incorrect indication after each answer
- **Correct Answer Display**: Shows correct answer when wrong
- **Visual Indicators**: Checkmark for correct, X for incorrect

### Copy Functionality
- **Right-Click Menus**: Copy question text, options, or full questions
- **Text Selection**: Copy selected text portions
- **Full Question Copy**: Copy complete question with all options and correct answer marked
- **Input Field Copy**: Copy entered text from calculation questions

### Results & Scoring
- **Score Display**: Fraction and percentage scores
- **Performance Categories**: Excellent (90%+), Great (80%+), Good (70%+), Fair (60%+), Review (<60%)
- **Time Display**: Shows completion time for Everything Test
- **Final Results**: Complete breakdown at quiz end

### User Interface
- **Dark Theme**: Modern dark interface with blue accents
- **Responsive Design**: Window resizing support
- **Smooth Scrolling**: Mouse wheel support for long content
- **Clean Typography**: Clear, readable fonts throughout

### Technical Features
- **Question Persistence**: No question repetition tracking across sessions
- **Configuration Memory**: Settings preserved during quiz session
- **Error Handling**: Input validation and error messages
- **Smooth Performance**: Optimized rendering and updates

## Usage

1. **Start Application**: Run `python main.py`
2. **Select Quiz**: Choose from individual quizzes, Everything Test, or Random
3. **Configure Options**: Select question types and randomization preferences
4. **Take Quiz**: Answer questions using mouse clicks or keyboard
5. **Review Results**: See score, time, and performance evaluation
6. **Restart or Return**: Restart same quiz or return to main menu

## Keyboard Shortcuts
- **Space Bar**: Submit answer or continue to next question
- **Enter**: Submit answer for input/calculation questions

## Files Required
- `main.py`: Main application file
- `questions.py`: Question database with all quiz content
