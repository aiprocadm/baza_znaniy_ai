"""add api keys usage billing and rag run tables

Revision ID: 20260506_01_api_keys_usage_rag
Revises: 20260503_03_npa_fields
Create Date: 2026-05-06
"""
from alembic import op
import sqlalchemy as sa

revision = '20260506_01_api_keys_usage_rag'
down_revision = '20260503_03_npa_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('api_keys',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('key_hash', sa.String(), nullable=False),
        sa.Column('label', sa.String(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_api_keys_tenant_id', 'api_keys', ['tenant_id'])
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'], unique=True)
    for table in ('usage_events', 'billing_events'):
        op.create_table(table,
            sa.Column('id', sa.Integer(), primary_key=True),
            sa.Column('tenant_id', sa.String(), nullable=False),
            sa.Column('subject_type', sa.String(), nullable=False),
            sa.Column('subject_id', sa.String(), nullable=False),
            sa.Column('event_type', sa.String(), nullable=False),
            sa.Column('idempotency_key', sa.String(), nullable=True),
            sa.Column('payload', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            *([sa.Column('amount', sa.Float(), nullable=False), sa.Column('currency', sa.String(), nullable=False)] if table == 'billing_events' else []),
        )
    op.create_table('rag_runs',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('tenant_id', sa.String(), nullable=False),
        sa.Column('subject_type', sa.String(), nullable=False),
        sa.Column('subject_id', sa.String(), nullable=False),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_table('rag_run_sources',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('rag_run_id', sa.Integer(), nullable=False),
        sa.Column('source_file', sa.String(), nullable=True),
        sa.Column('source_page', sa.Integer(), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table('rag_run_sources')
    op.drop_table('rag_runs')
    op.drop_table('billing_events')
    op.drop_table('usage_events')
    op.drop_index('ix_api_keys_key_hash', table_name='api_keys')
    op.drop_index('ix_api_keys_tenant_id', table_name='api_keys')
    op.drop_table('api_keys')
