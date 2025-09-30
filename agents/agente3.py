import io
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

def agente3_formatar_apresentacao(resultado_texto, resultado_df, pergunta, img_bytes):
    """Gera o relatório em PDF (ReportLab)."""
    
    pdf_bytes = None
    final_text_output = resultado_texto

    # Usa o DataFrame para texto se estiver disponível (para PDF)
    if resultado_df is not None and not resultado_df.empty:
        final_text_output = resultado_df.to_markdown(index=resultado_df.index.name is not None)
    
    try:
        pdf_output_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_output_buffer, pagesize=A4)
        width, height = A4
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, height - inch, "Relatório de Análise de Dados")
        
        c.setFont("Helvetica", 12)
        textobject = c.beginText(inch, height - inch - 20)
        textobject.textLines(f"Pergunta: {pergunta}")
        c.drawText(textobject)
        
        y_pos = height - inch - 50

        if resultado_df is not None and not resultado_df.empty:
            
            # Adiciona o índice como primeira coluna se o DataFrame tiver um nome de índice definido
            if resultado_df.index.name is not None:
                data = [
                    [resultado_df.index.name] + resultado_df.columns.tolist()
                ] + [
                    [str(idx)] + row for idx, row in zip(resultado_df.index.tolist(), resultado_df.values.astype(str).tolist())
                ]
            else:
                data = [resultado_df.columns.tolist()] + resultado_df.values.astype(str).tolist()

            # Limita a largura da tabela
            max_cols = 10 
            data = [row[:max_cols] for row in data]
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            table_width, table_height = table.wrapOn(c, width, height)
            y_pos -= table_height + 20
            
            if y_pos < inch:
                c.showPage()
                y_pos = height - inch - 20
            
            table.drawOn(c, inch, y_pos)
            y_pos -= 20
        
        if img_bytes:
            image_reader = ImageReader(io.BytesIO(img_bytes))
            img_width, img_height = image_reader.getSize()
            
            aspect_ratio = img_width / img_height
            max_width = width - 2 * inch
            
            img_width = max_width
            img_height = img_width / aspect_ratio
            
            y_pos -= img_height + 20
            
            if y_pos < inch:
                c.showPage()
                y_pos = height - inch - 20
            
            c.drawImage(image_reader, inch, y_pos, width=img_width, height=img_height)

        c.save()
        pdf_bytes = pdf_output_buffer.getvalue()

    except Exception as e:
        pdf_bytes = None

    return final_text_output, img_bytes, pdf_bytes