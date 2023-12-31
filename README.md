# EMI
#### Background: 
From 2021, the project of a fully English-taught course started at NSYSU. Every student gives feedback on teaching and rates their own learning at the end of each semester. There are 20 questions for each, scored from 1 to 7. Higher scores mean a better view of the teacher or more dedication to the subject.

#### Target: 
Measuring achievements through teaching satisfaction and self-assessment of study performance in a fully English-taught course to examine the 'customer acceptance' of fully English classes.

#### Models: 
Kolmogorov Smirnov Test(Scipy)

## Introdution 
- Why I choose K-S test: I used the K-S test to check if the data follows a normal distribution. The obtained p-values reject this idea, proving that the data is not normally distributed. Since I can't use a t-test, I'll use non-parametric tests to evaluate the effectiveness of EMI.
![Figure 2022-06-25 175318 (0)](https://github.com/WSY-Samuel/EMI/assets/87291914/158368c9-3365-4b32-bdec-2f6e5b63944e)
![Figure 2022-06-25 175318 (1)](https://github.com/WSY-Samuel/EMI/assets/87291914/d5ed6e62-0fe1-4a10-abad-445ada94edfd)  
![Figure 2022-06-25 175318 (2)](https://github.com/WSY-Samuel/EMI/assets/87291914/aee50772-ba4e-425c-a1d4-584f4a5d9e85)
![Figure 2022-06-25 175318 (3)](https://github.com/WSY-Samuel/EMI/assets/87291914/7c0e673e-8006-4f5b-9ea7-187e33e339c5)  

- The primary analysis covers the entire university,the Chemistry, Electrical Engineering, and Mechatronics Engineering departments, focusing on the average student satisfaction and self-assessed learning outcomes for courses taught in both Chinese and English.

## Conclusion:
1. Students rate their learning outcomes differently between courses in Chinese and English across the whole university.
2. In teaching satisfaction, there's a notable difference between fully English-taught programs and regular programs in Electrical Engineering and Mechatronics Engineering.
3. Both Chinese-English courses and fully English-taught programs score well above 5, mostly exceeding 6, for teaching satisfaction and learning outcomes.
4. Teachers generally receive high satisfaction scores for both teaching and effectiveness.  
**Most instructors have scores concentrated above 5, with only a few falling below. This suggests that students are generally highly satisfied and positive about most instructors' teaching effectiveness.**

![Figure 2022-06-27 124850 (0)](https://github.com/WSY-Samuel/EMI/assets/87291914/8e53df4c-b09a-426c-a672-e35194afafd2)
![Figure 2022-06-27 124850 (1)](https://github.com/WSY-Samuel/EMI/assets/87291914/08affeb0-34af-4c41-80d5-320d8353787f)  
![Figure 2022-06-27 132542 (0)](https://github.com/WSY-Samuel/EMI/assets/87291914/a2f70d66-2964-4f9a-a680-aadf4bff527a)
![Figure 2022-06-27 132542 (1)](https://github.com/WSY-Samuel/EMI/assets/87291914/61f6a038-1786-4b56-975b-f99d2272d165)  

### The reason why lack of signifiance:
I believe that the lack of significance may be due to the close average satisfaction scores in both Chinese and English courses, with a ceiling effect (maximum score of 7) potentially limiting the observable differences.
