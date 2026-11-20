"""Rich HTML + plain-text Gmail SMTP notifier with human-friendly verdict banners."""

from __future__ import annotations

import html
import inspect
import logging
import os
import smtplib
import ssl
from collections.abc import Callable, Sequence
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any
from zoneinfo import ZoneInfo

from src.notifications.telegram import TelegramNotifier

logger = logging.getLogger(__name__)
NY_TIMEZONE = ZoneInfo("America/New_York")


def _build_html_card(
    badge_bg: str,
    badge_text: str,
    headline: str,
    subtitle: str,
    rows: Sequence[tuple[str, str]],
    footer_note: str,
) -> str:
    """Render a clean, mobile-friendly HTML email card with a clear good-news/risk banner."""
    rows_html = "\n".join(
        f"""
        <tr>
          <td style="padding:10px 14px;border-bottom:1px solid #eef2f6;color:#475569;font-size:14px;">{html.escape(label)}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #eef2f6;color:#0f172a;font-weight:600;font-size:14px;text-align:right;">{html.escape(value)}</td>
        </tr>"""
        for label, value in rows
    )
    now_str = datetime.now(NY_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S ET")
    return f"""<!DOCTYPE html>
<html lang="es">
<body style="margin:0;padding:20px;background-color:#f8fafc;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;">
  <div style="max-width:560px;margin:0 auto;background:#ffffff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;box-shadow:0 2px 6px rgba(15,23,42,0.06);">
    <div style="background:{badge_bg};color:#ffffff;padding:18px 20px;">
      <div style="font-size:12px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;opacity:0.95;">{html.escape(badge_text)}</div>
      <div style="font-size:20px;font-weight:800;margin-top:4px;">{html.escape(headline)}</div>
      <div style="font-size:14px;margin-top:6px;opacity:0.95;line-height:1.4;">{html.escape(subtitle)}</div>
    </div>
    <table style="width:100%;border-collapse:collapse;">
      {rows_html}
    </table>
    <div style="padding:14px 18px;background:#f8fafc;color:#475569;font-size:13px;line-height:1.5;">
      <strong>¿Qué significa esto?</strong> {html.escape(footer_note)}
      <div style="margin-top:8px;color:#94a3b8;font-size:11px;">Trading Bot Pro · Guardián Algorítmico · {now_str}</div>
    </div>
  </div>
</body>
</html>"""


class EmailNotifier:
    """Fail-safe Gmail SMTP notifier sending executive HTML + plain-text trade cards."""

    def __init__(
        self,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        smtp_user: str | None = None,
        smtp_app_password: str | None = None,
        recipient_email: str | None = None,
        timeout: float = 15.0,
        smtp_factory: Callable[..., Any] | None = None,
    ) -> None:
        raw_host = os.getenv("SMTP_HOST", "smtp.gmail.com") if smtp_host is None else smtp_host
        raw_port = os.getenv("SMTP_PORT", "587") if smtp_port is None else str(smtp_port)
        raw_user = os.getenv("SMTP_USER", "") if smtp_user is None else smtp_user
        raw_pass = (
            os.getenv("SMTP_APP_PASSWORD", "")
            if smtp_app_password is None
            else smtp_app_password
        )
        raw_to = (
            os.getenv("EMAIL_RECIPIENT", raw_user)
            if recipient_email is None
            else recipient_email
        )

        self.smtp_host: str = raw_host.strip() or "smtp.gmail.com"
        self.smtp_port: int = int(raw_port) if str(raw_port).strip().isdigit() else 587
        self.smtp_user: str = raw_user.strip()
        self.smtp_app_password: str = raw_pass.strip()
        self.recipient_email: str = raw_to.strip() or self.smtp_user
        self.timeout: float = float(timeout)
        self._smtp_factory = smtp_factory
        self._veto_dedup_cache: set[tuple[str, str]] = set()
        self._warned_unconfigured: bool = False

    @property
    def is_configured(self) -> bool:
        """Return True when SMTP user, app password, and recipient are present."""
        return bool(self.smtp_user and self.smtp_app_password and self.recipient_email)

    def send_email(
        self,
        subject: str,
        plain_body: str,
        html_body: str | None = None,
    ) -> bool:
        """Send a multipart HTML + plain-text email fail-safely without raising."""
        if not self.is_configured:
            if not self._warned_unconfigured:
                logger.warning(
                    "Email notification skipped: SMTP_USER, SMTP_APP_PASSWORD, or EMAIL_RECIPIENT is empty"
                )
                self._warned_unconfigured = True
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject.strip()
        msg["From"] = self.smtp_user
        msg["To"] = self.recipient_email
        msg.attach(MIMEText(plain_body.strip(), "plain", "utf-8"))
        if html_body:
            msg.attach(MIMEText(html_body, "html", "utf-8"))

        try:
            factory = self._smtp_factory or smtplib.SMTP
            with factory(self.smtp_host, self.smtp_port, timeout=self.timeout) as server:
                if self._smtp_factory is None:
                    server.ehlo()
                    server.starttls(context=ssl.create_default_context())
                    server.ehlo()
                server.login(self.smtp_user, self.smtp_app_password)
                server.sendmail(self.smtp_user, [self.recipient_email], msg.as_string())
            return True
        except Exception as exc:
            logger.warning("Email notification failed (fail-safe): %s", exc)
            return False

    def send_message(self, text: str) -> bool:
        """Send a generic system/error alert email."""
        cleaned = text.strip()
        if not cleaned:
            return False
        html_body = _build_html_card(
            badge_bg="#d97706",
            badge_text="AVISO DEL SISTEMA",
            headline="Notificación Operativa del Bot",
            subtitle=cleaned,
            rows=[("Estado", "Bot en ejecución con protección activa")],
            footer_note="Este aviso informa un evento técnico puntual; las posiciones abiertas siguen protegidas por sus órdenes Stop-Limit GTC.",
        )
        return self.send_email(
            subject="⚠️ [Trading Bot] Aviso Operativo",
            plain_body=cleaned,
            html_body=html_body,
        )

    def notify_trade_executed(
        self,
        symbol: str,
        side: str,
        qty: float | int,
        limit_price: float,
        stop_price: float | None = None,
        dry_run: bool = False,
    ) -> bool:
        """Send a 'Good News' card when a new Limit buy passes all 5 funnel stages."""
        sym = symbol.strip().upper()
        mode_tag = " [SIMULACIÓN DRY-RUN]" if dry_run else ""
        total_invested = float(qty) * float(limit_price)
        risk_str = (
            f"${stop_price:,.2f} (riesgo máx. ${(limit_price - stop_price) * float(qty):,.2f})"
            if stop_price is not None and stop_price < limit_price
            else (f"${stop_price:,.2f}" if stop_price is not None else "N/A")
        )
        rows = [
            ("Acción (Ticker)", sym),
            ("Tipo de Orden", f"{side.strip().upper()} LIMIT (GTC){mode_tag}"),
            ("Cantidad", f"{qty} acciones"),
            ("Precio Límite", f"${limit_price:,.2f} USD"),
            ("Inversión Total", f"${total_invested:,.2f} USD"),
            ("Stop-Loss Protector (GTC)", risk_str),
        ]
        html_body = _build_html_card(
            badge_bg="#16a34a",
            badge_text=f"🟢 BUENA NOTICIA · NUEVA INVERSIÓN APROBADA{mode_tag}",
            headline=f"Compra de {sym} Ejecutada (${total_invested:,.2f})",
            subtitle=f"{sym} superó las 5 etapas: Liquidez, Tendencia Alcista, Analista IA y Guardián de Riesgo.",
            rows=rows,
            footer_note="Es una señal positiva: el bot encontró una oportunidad con respaldo institucional y ya colocó su escudo Stop-Loss GTC automático.",
        )
        plain = (
            f"BUENA NOTICIA - NUEVA INVERSION APROBADA{mode_tag}\n"
            f"Accion: {sym} | {side.upper()} {qty} acc. @ ${limit_price:,.2f} "
            f"(Total: ${total_invested:,.2f})\n"
            f"Stop-Loss GTC: {risk_str}"
        )
        return self.send_email(
            subject=f"🟢 [Buena Noticia] Compra aprobada en {sym} (${total_invested:,.2f}){mode_tag}",
            plain_body=plain,
            html_body=html_body,
        )

    def notify_position_closed(
        self,
        symbol: str,
        action: str,
        qty: float | int,
        limit_price: float | None,
        reason: str,
        entry_price: float | None = None,
        dry_run: bool = False,
    ) -> bool:
        """Send an executive exit card clearly stating whether the trade won or protected capital."""
        sym = symbol.strip().upper()
        act = action.strip().upper()
        mode_tag = " [DRY-RUN]" if dry_run else ""
        exit_p = float(limit_price) if limit_price is not None else 0.0
        pnl_dollars: float | None = None
        pnl_pct: float | None = None
        if entry_price is not None and entry_price > 0.0 and exit_p > 0.0:
            pnl_dollars = (exit_p - float(entry_price)) * float(qty)
            pnl_pct = ((exit_p - float(entry_price)) / float(entry_price)) * 100.0

        is_win = (pnl_dollars is not None and pnl_dollars > 0.0) or act in {
            "TAKE_PROFIT_EXIT",
            "PEAK_EXHAUSTION_EXIT",
        }
        is_emergency = act == "EMERGENCY_EXIT"

        if is_win:
            badge_bg = "#16a34a"
            badge_text = f"🟢 ¡EXCELENTE NOTICIA! · GANANCIA REALIZADA{mode_tag}"
            pnl_headline = (
                f"+${pnl_dollars:,.2f} USD ({pnl_pct:+.2f}%)"
                if pnl_dollars is not None and pnl_pct is not None
                else "Objetivo de Ganancia Alcanzado"
            )
            headline = f"Venta Ganadora en {sym}: {pnl_headline}"
            subtitle = "El bot cerró la posición asegurando ganancias reales en tu cuenta."
            footer = "¡Es una muy buena noticia! El capital más la ganancia ya quedaron libres en efectivo para la siguiente oportunidad."
            subj_prefix = f"🟢 [¡Ganancia en {sym}!] {pnl_headline}"
        elif is_emergency:
            badge_bg = "#dc2626"
            badge_text = f"🔴 PROTECCIÓN DE EMERGENCIA · SALIDA RÁPIDA{mode_tag}"
            pnl_headline = (
                f"${pnl_dollars:+,.2f} USD ({pnl_pct:+.2f}%)"
                if pnl_dollars is not None and pnl_pct is not None
                else "Cierre Preventivo"
            )
            headline = f"Escudo Anti-Caída Activado en {sym} ({pnl_headline})"
            subtitle = "Se detectó una caída anormal en los primeros minutos y el bot protegió tu capital."
            footer = "Aunque cerró en negativo, el escudo evitó una pérdida mayor ante un movimiento brusco del mercado."
            subj_prefix = f"🔴 [Escudo Emergencia] Salida protectora en {sym}"
        else:
            badge_bg = "#ca8a04"
            badge_text = f"🟡 SALIDA PROTECTORA · CAPITAL RESGUARDADO{mode_tag}"
            pnl_headline = (
                f"${pnl_dollars:+,.2f} USD ({pnl_pct:+.2f}%)"
                if pnl_dollars is not None and pnl_pct is not None
                else act
            )
            headline = f"Cierre Controlado en {sym} ({pnl_headline})"
            subtitle = "El Stop Dinámico o límite de tiempo cerró la operación dentro del riesgo planeado."
            footer = "Es una salida disciplinada: en trading profesional cortar a tiempo protege tu cuenta y libera efectivo para mejores acciones."
            subj_prefix = f"🟡 [Capital Protegido] Cierre controlado en {sym} ({pnl_headline})"

        rows: list[tuple[str, str]] = [
            ("Acción (Ticker)", sym),
            ("Motivo de Salida", act),
            ("Cantidad Vendida", f"{qty} acciones"),
        ]
        if entry_price is not None and entry_price > 0.0:
            rows.append(("Precio de Entrada", f"${entry_price:,.2f} USD"))
        rows.append(("Precio de Salida", f"${exit_p:,.2f} USD" if exit_p > 0 else "N/A"))
        if pnl_dollars is not None and pnl_pct is not None:
            rows.append(("Resultado Neto (PnL)", f"${pnl_dollars:+,.2f} USD ({pnl_pct:+.2f}%)"))
        rows.append(("Detalle Técnico", reason))

        html_body = _build_html_card(
            badge_bg=badge_bg,
            badge_text=badge_text,
            headline=headline,
            subtitle=subtitle,
            rows=rows,
            footer_note=footer,
        )
        plain = (
            f"{badge_text}\n{headline}\n"
            f"Ticker: {sym} | Cant: {qty} | Salida: ${exit_p:,.2f}\n"
            f"Detalle: {reason}"
        )
        return self.send_email(
            subject=f"{subj_prefix}{mode_tag}",
            plain_body=plain,
            html_body=html_body,
        )

    def notify_guardian_veto(
        self,
        symbol: str,
        veto_reasons: Sequence[str],
        vix: float | None = None,
    ) -> bool:
        """Send at most one protective veto digest per symbol per NY day to avoid inbox spam."""
        sym = symbol.strip().upper()
        ny_day = datetime.now(NY_TIMEZONE).strftime("%Y-%m-%d")
        dedup_key = (ny_day, sym)
        if dedup_key in self._veto_dedup_cache:
            return False
        self._veto_dedup_cache.add(dedup_key)

        reasons_joined = "; ".join(veto_reasons) or "Regla de riesgo activada"
        rows = [
            ("Acción Evaluada", sym),
            ("Decisión del Guardián", "COMPRA BLOQUEADA (Veto Protector)"),
            ("Índice de Volatilidad (VIX)", f"{vix:.2f}" if vix is not None else "N/A"),
            ("Razón del Bloqueo", reasons_joined),
        ]
        html_body = _build_html_card(
            badge_bg="#2563eb",
            badge_text="🛡️ BUENA NOTICIA DE PROTECCIÓN · RIESGO EVITADO",
            headline=f"El Guardián Evitó Comprar {sym} Hoy",
            subtitle="La IA vio interés en la acción, pero el Guardián matemático detectó un riesgo y protegió tu dinero.",
            rows=rows,
            footer_note="Que el Guardián bloquee una compra cerca de reportes trimestrales (Earnings), alta volatilidad o sobreexposición es una buena noticia: cuida tu patrimonio.",
        )
        plain = f"GUARDIAN VETO PROTECTOR ({sym})\nRazon: {reasons_joined}"
        return self.send_email(
            subject=f"🛡️ [Riesgo Evitado] El Guardián protegió tu capital en {sym}",
            plain_body=plain,
            html_body=html_body,
        )

    def notify_circuit_breaker(
        self,
        drawdown_pct: float,
        starting_equity: float,
        daily_pnl: float,
    ) -> bool:
        """Send a high-priority alert when the daily drawdown circuit breaker halts new buys."""
        rows = [
            ("Retroceso del Día", f"{drawdown_pct:.2%}"),
            ("Capital al Inicio del Día", f"${starting_equity:,.2f} USD"),
            ("Variación del Día (PnL)", f"${daily_pnl:+,.2f} USD"),
            ("Acción Tomada", "Nuevas compras pausadas por hoy"),
        ]
        html_body = _build_html_card(
            badge_bg="#dc2626",
            badge_text="🔴 ALERTA DE PROTECCIÓN MÁXIMA · CIRCUIT BREAKER",
            headline="Pausa Preventiva de Compras por Hoy",
            subtitle="El mercado tuvo un retroceso diario superior al límite del 3% y el Guardián congeló nuevas compras.",
            rows=rows,
            footer_note="Tus posiciones actuales siguen protegidas por sus órdenes Stop-Limit GTC. Mañana el sistema reevaluará el mercado normalmente.",
        )
        plain = (
            f"CIRCUIT BREAKER ACTIVADO\n"
            f"Drawdown diario: {drawdown_pct:.2%} | PnL: ${daily_pnl:+,.2f} | Equity base: ${starting_equity:,.2f}"
        )
        return self.send_email(
            subject=f"🔴 [Circuit Breaker] Pausa protectora diaria activada ({drawdown_pct:.2%})",
            plain_body=plain,
            html_body=html_body,
        )

    def notify_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        cash: float | None = None,
        open_positions: Sequence[Any] = (),
    ) -> bool:
        """Send an executive end-of-day account summary card."""
        base_eq = equity - daily_pnl
        daily_pct = (daily_pnl / base_eq * 100.0) if base_eq > 0.0 else 0.0
        is_positive = daily_pnl >= 0.0
        badge_bg = "#16a34a" if is_positive else "#2563eb"
        badge_text = (
            "🟢 RESUMEN DIARIO · BALANCE POSITIVO"
            if is_positive
            else "🛡️ RESUMEN DIARIO · CIERRE DE JORNADA"
        )
        headline = f"Balance del Día: ${daily_pnl:+,.2f} USD ({daily_pct:+.2f}%)"
        subtitle = (
            "Resumen ejecutivo de tu cuenta al cierre de la sesión de Nueva York."
        )
        pos_labels: list[str] = []
        for pos in open_positions:
            sym = str(getattr(pos, "symbol", "") or "").strip().upper()
            shares = getattr(pos, "shares", getattr(pos, "qty", 0))
            if sym:
                pos_labels.append(f"{sym} ({shares})")
        positions_str = ", ".join(pos_labels) if pos_labels else "0 (100% en efectivo)"

        rows: list[tuple[str, str]] = [
            ("Valor Total de la Cuenta (Equity)", f"${equity:,.2f} USD"),
            ("Resultado del Día (PnL)", f"${daily_pnl:+,.2f} USD ({daily_pct:+.2f}%)"),
        ]
        if cash is not None:
            rows.append(("Efectivo Disponible (Cash)", f"${cash:,.2f} USD"))
        rows.append(("Posiciones Abiertas ({})".format(len(pos_labels)), positions_str))

        footer = (
            "Todas las posiciones abiertas quedan resguardadas durante la noche con órdenes Stop-Limit GTC activas en el bróker."
        )
        html_body = _build_html_card(
            badge_bg=badge_bg,
            badge_text=badge_text,
            headline=headline,
            subtitle=subtitle,
            rows=rows,
            footer_note=footer,
        )
        plain = (
            f"{badge_text}\n{headline}\n"
            f"Equity: ${equity:,.2f} | Cash: ${cash or 0.0:,.2f} | Posiciones: {positions_str}"
        )
        icon = "🟢" if is_positive else "🛡️"
        return self.send_email(
            subject=f"{icon} [Resumen Diario] Equity ${equity:,.2f} ({daily_pnl:+,.2f} USD)",
            plain_body=plain,
            html_body=html_body,
        )


class CompositeNotifier:
    """Dispatches notifications to EmailNotifier (primary) and TelegramNotifier (optional)."""

    def __init__(
        self,
        email_notifier: EmailNotifier | None = None,
        telegram_notifier: TelegramNotifier | None = None,
    ) -> None:
        self.email = email_notifier or EmailNotifier()
        self.telegram = telegram_notifier or TelegramNotifier()

    def send_message(self, text: str) -> bool:
        sent_email = self.email.send_message(text) if self.email.is_configured else False
        sent_tg = self.telegram.send_message(text) if self.telegram.is_configured else False
        if not self.email.is_configured and not self.telegram.is_configured:
            self.email.send_message(text)
        return sent_email or sent_tg

    def notify_trade_executed(
        self,
        symbol: str,
        side: str,
        qty: float | int,
        limit_price: float,
        stop_price: float | None = None,
        dry_run: bool = False,
    ) -> bool:
        sent_email = (
            self.email.notify_trade_executed(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=limit_price,
                stop_price=stop_price,
                dry_run=dry_run,
            )
            if self.email.is_configured
            else False
        )
        sent_tg = (
            self.telegram.notify_trade_executed(
                symbol=symbol,
                side=side,
                qty=qty,
                limit_price=limit_price,
                stop_price=stop_price,
                dry_run=dry_run,
            )
            if self.telegram.is_configured
            else False
        )
        return sent_email or sent_tg

    def notify_position_closed(
        self,
        symbol: str,
        action: str,
        qty: float | int,
        limit_price: float | None,
        reason: str,
        entry_price: float | None = None,
        dry_run: bool = False,
    ) -> bool:
        sent_email = (
            self.email.notify_position_closed(
                symbol=symbol,
                action=action,
                qty=qty,
                limit_price=limit_price,
                reason=reason,
                entry_price=entry_price,
                dry_run=dry_run,
            )
            if self.email.is_configured
            else False
        )
        sent_tg = (
            self.telegram.notify_position_closed(
                symbol=symbol,
                action=action,
                qty=qty,
                limit_price=limit_price,
                reason=reason,
            )
            if self.telegram.is_configured
            else False
        )
        return sent_email or sent_tg

    def notify_guardian_veto(
        self,
        symbol: str,
        veto_reasons: Sequence[str],
        vix: float | None = None,
    ) -> bool:
        sent_email = (
            self.email.notify_guardian_veto(
                symbol=symbol,
                veto_reasons=veto_reasons,
                vix=vix,
            )
            if self.email.is_configured
            else False
        )
        sent_tg = (
            self.telegram.notify_guardian_veto(
                symbol=symbol,
                veto_reasons=veto_reasons,
                vix=vix,
            )
            if self.telegram.is_configured
            else False
        )
        return sent_email or sent_tg

    def notify_circuit_breaker(
        self,
        drawdown_pct: float,
        starting_equity: float,
        daily_pnl: float,
    ) -> bool:
        sent_email = (
            self.email.notify_circuit_breaker(
                drawdown_pct=drawdown_pct,
                starting_equity=starting_equity,
                daily_pnl=daily_pnl,
            )
            if self.email.is_configured
            else False
        )
        sent_tg = (
            self.telegram.notify_circuit_breaker(
                drawdown_pct=drawdown_pct,
                starting_equity=starting_equity,
                daily_pnl=daily_pnl,
            )
            if self.telegram.is_configured
            else False
        )
        return sent_email or sent_tg

    def notify_daily_summary(
        self,
        equity: float,
        daily_pnl: float,
        cash: float | None = None,
        open_positions: Sequence[Any] = (),
    ) -> bool:
        sent_email = (
            self.email.notify_daily_summary(
                equity=equity,
                daily_pnl=daily_pnl,
                cash=cash,
                open_positions=open_positions,
            )
            if self.email.is_configured
            else False
        )
        return sent_email
