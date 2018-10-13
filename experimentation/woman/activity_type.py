from enum import Enum


class ActivityType(Enum):
    BEGIN_PROCESS = 'begin_of_process'
    BEGIN_ACTIVITY = 'begin_of_activity'
    END_PROCESS = 'end_of_process'
    END_ACTIVITY = 'end_of_activity'

