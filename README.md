# EMI
#### Background: 
From 2021, the project of a fully English-taught course started at NSYSU. Every student gives feedback on teaching and rates their own learning at the end of each semester. There are 20 questions for each, scored from 1 to 7. Higher scores mean a better view of the teacher or more dedication to the subject.

#### Target: 
Measuring achievements through teaching satisfaction and self-assessment of study performance in a fully English-taught course to examine the 'customer acceptance' of fully English classes.

#### Models: 
kolmogorov smirnov test(Scipy)

## Introdution 
- Why I choose K-S test: I used the K-S test to check if the data follows a normal distribution. The obtained p-values reject this idea, proving that the data is not normally distributed. Since I can't use a t-test, I'll use non-parametric tests to evaluate the effectiveness of EMI.
- The primary analysis covers the entire university,the Chemistry, Electrical Engineering, and Mechatronics Engineering departments, focusing on the average student satisfaction and self-assessed learning outcomes for courses taught in both Chinese and English.

## Conclusion:
1. Students rate their learning outcomes differently between courses in Chinese and English across the whole university.
2. In teaching satisfaction, there's a notable difference between fully English-taught programs and regular programs in Electrical Engineering and Mechatronics Engineering.
3. Both Chinese-English courses and fully English-taught programs score well above 5, mostly exceeding 6, for teaching satisfaction and learning outcomes.
4. Teachers generally receive high satisfaction scores for both teaching and effectiveness.  
**Most instructors have scores concentrated above 5, with only a few falling below. This suggests that students are generally highly satisfied and positive about most instructors' teaching effectiveness.**

## The reason why lack of signifiance:
I believe that the lack of significance may be due to the close average satisfaction scores in both Chinese and English courses, with a ceiling effect (maximum score of 7) potentially limiting the observable differences.
