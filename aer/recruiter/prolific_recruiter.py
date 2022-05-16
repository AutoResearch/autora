from typing import Any, List
import requests
from dotenv import load_dotenv
import os
from .study_options import DeviceOptions, PeripheralOptions, ProlificIdOptions, CompletionOptions, StudyAction

class ProlificRecruiter():
    """
    Handles interaction with prolific (study creation, publishing, etc.)
    Documentation: https://docs.prolific.co/docs/api-docs/public/
    """
    API_VERSION = 'v1/'
    BASE_URL = f'https://api.prolific.co/api/{API_VERSION}'
    
    # currently using dotenv to store environment variables
    load_dotenv()
    
    # currently, they only require an Authorization header
    # referer header is optional
    if "API_TOKEN" in os.environ:
        HEADERS = {'Authorization': f'Token {os.environ["API_TOKEN"]}'}
    else:
        HEADERS = {}
    
    def list_studies(self) -> Any:
        """
        Returns list of all studies on Prolific account.
        """
        studies = requests.get(f'{self.BASE_URL}studies/', headers=self.HEADERS)
        return studies.json()
    
    def draft_study(self,
                    name: str,
                    description: str,
                    external_study_url: str,
                    prolific_id_option: ProlificIdOptions,
                    completion_code: str, completion_option: CompletionOptions,
                    total_available_places: int, reward: int,
                    estimated_completion_time: int, 
                    maximum_allowed_time: int = None,
                    eligibility_requirements=[], 
                    device_compatibility: List[DeviceOptions] = [],
                    peripheral_requirements: List[PeripheralOptions] = [],
                    internal_name: str = None) -> bool:
        """
        Allows for a study to be drafted given the following parameters.

        Args:
            name (str): Name that will be displayed on prolific
            description (str): Description of study for participants
            external_study_url (str): URL to experiment website
            prolific_id_option (ProlificIdOptions): Method of collecting subject ID
            completion_code (str): Code subject uses to mark experiment completion
            completion_option (CompletionOptions): Method of signifying participation
            total_available_places (int): Participant limit
            estimated_completion_time (int): How long the study takes
            reward (int): Amount of payment for completion
            maximum_allowed_time (int, optional): Allows specification of time before submission times out.
            eligibility_requirements (list, optional): Allows various options to filter participants. Defaults to [] (no requirements).
            device_compatibility (list[DeviceOptions], optional): Allows selecting required devices. Defaults to [] (any device).
            peripheral_requirements (list[PeripheralOptions], optional): Allows specifying additional requirements. Defaults to [] (no other requirements).
            internal_name (str, optional): Private name of study. Defaults to None.
        
        Returns:
            bool: Whether the request was successful or not
        """
        # packages function parameters into dictionary
        data = locals()
        # removes self variable
        del locals['self']
        
        # removes optional parameters that aren't specified
        if maximum_allowed_time is None:
            del locals['maximum_allowed_time']
        if internal_name is None:
            del locals['internal_name']
        
        study = requests.post(f'{self.BASE_URL}studies/', headers=self.HEADERS, 
                              json=data)
        
        # handles request failure
        if study.status_code >= 400:
            print(study.json())
            return False
        return True
    
    def retrieve_study(self, study_id: str) -> Any:
        """
        Retrieves information about study given its ID.
        """
        study = requests.get(f'{self.BASE_URL}studies/{study_id}/')
        return study.json()
    
    def update_study(self, study_id: str, **kwargs) -> bool:
        """
        Updates the parameters of a given study.
        If a study is already published, only internal_name
        and total_available_places can be updated.
        """
        study = requests.patch(f'{self.BASE_URL}studies/{study_id}/', 
                               headers=self.HEADERS, json=kwargs)
        return study.status_code < 400
    
    def publish_study(self, study_id: str, 
                      action: StudyAction = StudyAction.PUBLISH) -> bool:
        """
        Performs action on specified study. Default action is to publish
        the study.
        """
        data = {"action": action}
        study = requests.post(f'{self.BASE_URL}studies/{study_id}/transition/', 
                              headers=self.HEADERS, json=data)
        return study.status_code < 400
