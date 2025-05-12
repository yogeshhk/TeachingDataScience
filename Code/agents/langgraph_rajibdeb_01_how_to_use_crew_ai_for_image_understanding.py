"""
Author: Rajib Deb
Date: 02/10/2024
Description: This is the driver program that starts the MasterChef crew.
"""
from crew_ai_crews.master_chef import MasterChefCrew

if __name__ == "__main__":
    # url="https://rumkisgoldenspoon.com/wp-content/uploads/2022/02/Aar-macher-jhol.jpg"
    url="https://m.media-amazon.com/images/I/51dFvTRE3iL.__AC_SX300_SY300_QL70_FMwebp_.jpg"
    crew = MasterChefCrew(url=url)
    print(crew.kickoff())