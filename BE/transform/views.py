#from .utils import generate_text as generate_bert, evaluate_model as evaluate_bert
from .utils1 import generate_text as generate_character, evaluate_model as evaluate_character
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def finish_text(request):
    try:
        prompt = request.GET.get('prompt', 'The')
        model_type = request.GET.get('model', 'bert')
        
        if model_type == 'character':
            response = generate_character(prompt)
        else:
            response = generate_bert(prompt)
            
        return Response({'generated_text': response, 'model_used': model_type})
    except FileNotFoundError as e:
        return Response({'error': f"Model not found: {str(e)}. Train the {model_type} model first."}, status=500)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['GET'])
def evaluate(request):
    try:
        model_type = request.GET.get('model', 'bert')
        
        if model_type == 'character':
            result = evaluate_character()
        else:
            result = evaluate_bert()
            
        return Response({
            "perplexity": result['perplexity'],
            "loss": result['loss'],
            "model": model_type
        })
    except FileNotFoundError as e:
        return Response({'error': f"Model not found: {str(e)}"}, status=500)
    except Exception as e:
        return Response({'error': str(e)}, status=500)