# forms.py 
from django import forms 
from .models import *

class CandiForm(forms.ModelForm): 

	class Meta: 
		model = Candi 
		fields = [
			'candi_url',
			#'candi_img'
				] 
