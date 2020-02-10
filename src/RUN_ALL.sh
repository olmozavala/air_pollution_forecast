#!/bin/bash

# READ!!!!!!! You may have to run the exports manually, not sure what is not working automatically
OZ_LIB="/home/olmozavala/Dropbox/MyProjects/OZ_LIB"
export PYTHONPATH="${PYTHONPATH}:$OZ_LIB/AI_Common/"
export PYTHONPATH="${PYTHONPATH}:$OZ_LIB/image_visualization/"
echo $PYTHONPATH

MAIN_CONFIG='/home/olmozavala/Dropbox/MyProjects/UNAM/PollutionMexicCity/PollutionForecastOZ/src/conf'
SRC_PATH='/home/olmozavala/Dropbox/MyProjects/UNAM/PollutionMexicCity/PollutionForecastOZ/src'

#["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY", "CUA"
#         ,"CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", "LAG", "LLA"
#         ,"LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT", "SAG", "SFE"
#         ,"SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL"
#         , "XCH"]

for station in ACO AJM AJU ARA ATI AZC BJU CAM CCA CES CFE CHO COR COY CUA CUI CUT DIC EAJ EDL FAC FAN GAM HAN HGM IBM IMP INN IZT \
LAA LAG LLA LOM LPR LVI MCM MER MGH MIN MON MPA NET NEZ PED PER PLA POT SAG SFE SHA SJA SNT SUR TAC TAH TAX TEC TLA TLI \
TPN UAX UIZ UNM VAL VIF XAL  XCH; do
  for pollutant in cont_otres; do
      cp -f $MAIN_CONFIG/TrainingUserConfigurationGeneral.py $MAIN_CONFIG/TrainingUserConfiguration.py
      sed -i 's/STATION/'$station'/' $MAIN_CONFIG/TrainingUserConfiguration.py
      sed -i 's/POLLUTANT/'$pollutant'/' $MAIN_CONFIG/TrainingUserConfiguration.py
      python $SRC_PATH/4_Training.py
  done
done
