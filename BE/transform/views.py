from .utils1 import generate_text, evaluate_model
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
import json

@api_view(['GET'])  # Change from POST to GET
def finish_text(request):
    try:
        prompt = request.GET.get('prompt', 'The')
        response = generate_text(prompt)
        return Response({'generated_text': response})
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def evaluate(request):
    try:
        result = evaluate_model()
        return Response({
            "perplexity": result['perplexity'],
            "loss": result['loss'],
            "note": result['note']
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)