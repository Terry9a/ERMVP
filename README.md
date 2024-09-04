# ERMVP

The official implementation of CVPR2024 paper "ERMVP: Communication-Efficient and Collaboration-Robust Multi-Vehicle Perception in Challenging Environments".
![ERMVP_Overview](https://github.com/Terry9a/ERMVP/blob/main/image.png)

> **ERMVP: Communication-Efficient and Collaboration-Robust Multi-Vehicle Perception in Challenging Environments** <br>
> Jingyu Zhang, Kun Yang, Yilei Wang, Hanqi Wang, Peng Sun\*, Liang Song\*<br>
> Accepted by CVPR2024

# Abstract

Collaborative perception enhances perception performance by enabling autonomous vehicles to exchange complementary information. Despite its potential to revolutionize the mobile industry, challenges in various environments, such as communication bandwidth limitations, localization errors and information aggregation inefficiencies, hinder its implementation in practical applications. In this work, we propose ERMVP, a communication-Efficient and collaboration-Robust Multi-Vehicle Perception method in challenging environments. Specifically, ERMVP has three distinct strengths: i) It utilizes the hierarchical feature sampling strategy to abstract a representative set of feature vectors, using less communication overhead for efficient communication; ii) It employs the sparse consensus features to execute precise spatial location calibrations, effectively mitigating the implications of vehicle localization errors; iii) A pioneering feature fusion and interaction paradigm is introduced to integrate holistic spatial semantics among different vehicles and data sources. To thoroughly validate our method, we conduct extensive experiments on real-world and simulated datasets. The results demonstrate that the proposed ERMVP is significantly superior to the state-of-the-art collaborative perception methods.

# Note

The code will be released after the publication of the subsequent work. You can refer to our previous work [Feaco](https://github.com/jmgu0212/FeaCo) on the basic implementation of the  feature spatial calibration module.
